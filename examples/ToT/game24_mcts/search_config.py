import copy
import re
from typing import Literal

import numpy as np
import scipy
import torch

from reasoners import SearchConfig, LanguageModel
from world_model import Game24State, Game24Action

from prompts.game24 import output_prompt, propose_prompt, value_prompt, value_last_step_prompt, value_map


class Game24Config(SearchConfig):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 n_actions=4,
                 batch_size=2,
                 depth_limit=4,
                 temperature=0.7,
                 n_eval=5,
                 calc_reward: Literal['sampling', 'logits'] = 'sampling') -> None:
        super().__init__()
        self.base_model = base_model
        self.example = None
        self.prompt = prompt
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.n_eval = n_eval
        self.value_cache = {}
        self.depth_limit = depth_limit
        self.temperature = temperature
        self.calc_reward = calc_reward

    @staticmethod
    def output_prompt_wrap(state: Game24State) -> str:
        return output_prompt.format(input=state.input, history='\n'.join(state.history))

    @staticmethod
    def propose_prompt_wrap(state: Game24State) -> str:
        return propose_prompt.format(input=state.current)

    @staticmethod
    def value_prompt_wrap(state: Game24State) -> str:
        return value_prompt.format(input=state.current)

    @staticmethod
    def value_last_step_prompt_wrap(state: Game24State) -> str:
        return value_last_step_prompt.format(input=state.input, answer=state.output)

    @staticmethod
    def retrieve_value(output: list[str]) -> float:
        # Take last paragraph, then last line (where modern LLMs put evaluations)
        keyword_counts = {'sure': 0, 'likely': 0, 'impossible': 0}
        
        output_names = []
        for text in output:
            # Get last paragraph
            last_paragraph = text.split('\n\n')[-1]
            # Get last line of last paragraph
            last_line = last_paragraph.split('\n')[-1]
            output_names.append(last_line)
        
        # Search for keywords in last lines (case insensitive)
        for text in output_names:
            text_lower = text.lower()
            for keyword in keyword_counts:
                if keyword in text_lower:
                    keyword_counts[keyword] += 1
        
        print(f'DEBUG: retrieve_value - output_names: {output_names}')
        print(f'DEBUG: retrieve_value - keyword_counts: {keyword_counts}')
        print(f'DEBUG: retrieve_value - value_map: {value_map}')
        value = sum(v * keyword_counts[k] for k, v in value_map.items())
        return value

    def get_actions(self, state: Game24State) -> list[Game24Action]:
        if state.current == '':
            return []
        print(f'DEBUG: Generating actions for state.current={repr(state.current)}')
        if state.current == '24':
            prompt = self.output_prompt_wrap(state)
            output = \
            self.base_model.generate([prompt], num_return_sequences=1, do_sample=False, eos_token_id='\n').text[0]
            output = 'Answer: ' + output.strip()
            print(f'DEBUG: Generated Answer action: {repr(output)}')
            return [output]
        elif ' ' not in state.current:
            return []
        else:
            prompt = self.propose_prompt_wrap(state)
            print(f'DEBUG: Prompt sent to model: {repr(prompt)}')
            output = \
            self.base_model.generate([prompt], num_return_sequences=1, do_sample=False, eos_token_id='Input').text[0]
            print(f'DEBUG: Raw model output: {repr(output)}')
            output = output.strip()
            # Don't split on \n\n as it removes the actual operations
            output = output.split('\n')
            print(f'DEBUG: Split lines: {output}')
            actions = [x for x in output if 'left' in x]
            # set does not guarantee order, but dict does guarantee
            # we cannot use set here because torch.distributed in LLaMA requires the same order across all processes
            actions = list(dict.fromkeys(actions))
            print(f'DEBUG: Generated {len(actions)} intermediate actions: {actions}')
            return actions

    def _reward(self, state: Game24State, action: Game24Action) -> float:
        if state.current == '':
            return 0.
        next_state = copy.deepcopy(state)
        if 'Answer' in action:
            match = re.match(r'Answer: (.*)', action)
            next_state.output = match[1] if match is not None else ''
        else:
            match = re.match(r'.*\(left: (.*)\)', action)
            next_state.current = match[1] if match is not None else ''
            next_state.history.append(action)

        print(f'DEBUG: _reward check - history_len={len(next_state.history)}, depth_limit={self.depth_limit}')
        if len(next_state.history) >= self.depth_limit:
            print(f'DEBUG: _reward returning 0.0 due to depth limit hit')
            return 0.
        if next_state.output is None:
            prompt = self.value_prompt_wrap(next_state)
            print(f'DEBUG: _reward using value_prompt for state.current={repr(next_state.current)}')
        else:
            prompt = self.value_last_step_prompt_wrap(next_state)
            print(f'DEBUG: _reward using value_last_step_prompt for output={repr(next_state.output)}')
        if prompt in self.value_cache:
            print(f'DEBUG: _reward found cached value={self.value_cache[prompt]}')
            return self.value_cache[prompt]
        print(f'DEBUG: _reward will evaluate with LLM, prompt={repr(prompt[:100])}...')

        if self.calc_reward == 'sampling':
            value_outputs = []
            for idx in range(0, self.n_eval, self.batch_size):
                n_samples = min(self.n_eval - idx, self.batch_size)
                output = self.base_model.generate([prompt], do_sample=True, temperature=self.temperature,
                                                  num_return_sequences=n_samples).text
                print(f'DEBUG: LLM raw output: {output}')
                processed_outputs = [o.strip() for o in output]  # Keep full text for retrieve_value()
                value_outputs += processed_outputs
                print(f'DEBUG: processed outputs: {processed_outputs}')
            print(f'DEBUG: all value_outputs: {value_outputs}')
            value = self.retrieve_value(value_outputs)
            print(f'DEBUG: retrieve_value result: {value}')
        elif self.calc_reward == 'logits':
            value_keys = list(value_map.keys())
            logits = self.base_model.get_next_token_logits([prompt], value_keys)[0]
            logits = scipy.special.softmax(logits)
            value = np.sum(logits * np.array(list(value_map.values())))
        else:
            raise NotImplementedError

        self.value_cache[prompt] = value
        # print(f'Reward of {state}, {action=} is {value:.5f}')
        return value

    def fast_reward(self, state: Game24State, action: Game24Action) -> tuple[float, dict]:
        reward = self._reward(state, action)
        print(f'DEBUG: fast_reward for action={repr(action[:30])}... = {reward}')
        return reward, {'reward': reward}

    # We calculate the full reward in fast_reward in Game24SearchConfig, direct return it
    def reward(self, state: Game24State, action: Game24Action, **kwargs) -> tuple[float, dict]:
        return self.fast_reward(state, action)
