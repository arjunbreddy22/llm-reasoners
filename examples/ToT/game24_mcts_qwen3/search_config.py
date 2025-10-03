import copy
import re
from typing import Literal, List

import numpy as np
import scipy
import torch
import time

from reasoners import SearchConfig, LanguageModel
from world_model import Game24State, Game24Action
import utils

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
        # Per-problem timing: first generate call to last generate return
        self.first_send_ts = None
        self.last_return_ts = None

    def _gen(self, *args, **kwargs):
        """Proxy to self.base_model.generate with timing for problem-level latency."""
        t0 = time.perf_counter()
        if self.first_send_ts is None:
            self.first_send_ts = t0
        out = self.base_model.generate(*args, **kwargs)
        self.last_return_ts = time.perf_counter()
        return out

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

    def _canonicalize_actions(self, state: Game24State, raw_text: str) -> List[Game24Action]:
        """Extract and canonicalize actions from raw model output.

        - Accepts lines anywhere in the text, even if surrounded by analysis.
        - Normalizes spacing and removes commas.
        - Reconstructs or validates the (left: ...) portion using the current multiset.
        - Deduplicates while preserving order.
        """
        # Normalize newlines and split for scanning
        text = raw_text.replace('\r\n', '\n').replace('\r', '\n')
        lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
        actions: List[str] = []

        # Regex for equations like: a op b = c
        eq_re = re.compile(r"^\s*(\d+)\s*([\+\-\*/])\s*(\d+)\s*=\s*(-?\d+(?:\.\d+)?)")
        left_re = re.compile(r"\(\s*left\s*:\s*([^\)]+)\)", re.IGNORECASE)

        # Helper to reconstruct (left: ...) using current numbers and the equation
        def reconstruct_left(equation: str) -> str:
            # Use utility to compute the left list deterministically
            return utils.correct_left_numbers(state.input, "\n".join(state.history), equation)

        seen = set()
        for ln in lines:
            # Skip any premature Answer lines here; answer is handled elsewhere
            if ln.lower().startswith('answer:'):
                continue
            m = eq_re.search(ln)
            if not m:
                continue
            a, op, b, c = m.groups()
            equation = f"{int(a)} {op} {int(b)} = {str(float(c)).rstrip('0').rstrip('.') if '.' in c else c}"

            # If a (left: ...) is present, validate; otherwise reconstruct
            m_left = left_re.search(ln)
            if m_left:
                # Reconstruct to canonicalize and validate
                canonical = reconstruct_left(equation)
            else:
                canonical = reconstruct_left(equation)

            # Normalize double spaces/commas
            canonical = canonical.replace(',', '').replace('  ', ' ').strip()
            if canonical not in seen:
                seen.add(canonical)
                actions.append(canonical)

        return actions

    def get_actions(self, state: Game24State) -> list[Game24Action]:
        if state.current == '':
            return []
        print(f'DEBUG: Generating actions for state.current={repr(state.current)}')
        if state.current == '24':
            prompt = self.output_prompt_wrap(state)
            output = self._gen([prompt], num_return_sequences=1, do_sample=False, eos_token_id='\n').text[0]
            output = 'Answer: ' + output.strip()
            print(f'DEBUG: Generated Answer action: {repr(output)}')
            return [output]
        elif ' ' not in state.current:
            print(f'DEBUG: Single number state {repr(state.current)} - returning [] (no actions possible)')
            return []
        else:
            # Strictly instruct the model to output only action lines
            base_prompt = self.propose_prompt_wrap(state)
            constraint_tail = (
                f"Output exactly {self.n_actions} lines.\n"
                "Each line must be of the form: A op B = C (left: X Y Z).\n"
                f"Use only the numbers from: {state.current}.\n"
                "Do not add any analysis, numbering, or extra text.\n"
            )
            prompt = base_prompt + constraint_tail
            print(f'DEBUG: Prompt sent to model: {repr(prompt)}')
            output = self._gen(
                [prompt],
                num_return_sequences=1,
                do_sample=False,
                eos_token_id='Input',
            ).text[0]
            print(f'DEBUG: Raw model output: {repr(output)}')
            actions = self._canonicalize_actions(state, output)
            # Cap to n_actions to avoid explosion
            actions = actions[: self.n_actions]
            print(f'DEBUG: Generated {len(actions)} intermediate actions: {actions}')
            return actions
        
        # This should never be reached, but add debug just in case
        print(f'DEBUG: get_actions fallthrough - returning [] for state.current={repr(state.current)}')
        return []

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
                # Force single-token, single-word style outputs when sampling
                strict_prompt = (
                    prompt
                    + "\nOutput exactly one word: sure | likely | impossible. Do not add any other text."
                )
                output = self._gen([strict_prompt], do_sample=False, temperature=0.0,
                                   num_return_sequences=n_samples, eos_token_id='\n').text
                print(f'DEBUG: LLM raw output: {output}')
                processed_outputs = [o.strip() for o in output]
                value_outputs += processed_outputs
                print(f'DEBUG: processed outputs: {processed_outputs}')
            print(f'DEBUG: all value_outputs: {value_outputs}')
            value = self.retrieve_value(value_outputs)
            print(f'DEBUG: retrieve_value result: {value}')
        elif self.calc_reward == 'logits':
            value_keys = list(value_map.keys())
            try:
                # Preferred path when backend exposes next-token logits for candidates
                logits = self.base_model.get_next_token_logits([prompt], value_keys)[0]
                probs = scipy.special.softmax(logits)
            except NotImplementedError:
                # For SGLang CompletionModel: use choice-based loglikelihood over label words
                contents = [prompt + k for k in value_keys]
                ll = self.base_model.get_loglikelihood(prompt, contents)
                # Convert log-likelihoods over choices into probabilities
                probs = scipy.special.softmax(ll)
            value = np.sum(probs * np.array(list(value_map.values())))
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
