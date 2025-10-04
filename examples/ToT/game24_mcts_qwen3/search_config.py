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
        # Normalize newlines and strip model meta-thinking tags
        text = raw_text.replace('\r\n', '\n').replace('\r', '\n')
        # Remove <think> blocks if present to surface the action lines
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL|re.IGNORECASE)
        text = text.replace("<think>", "").replace("</think>", "")
        lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
        actions: List[str] = []

        # Regex for equations like: a op b = c (non-anchored; allow integers or decimals for operands and result)
        number_pat = r"-?\d+(?:\.\d+)?"
        eq_re = re.compile(rf"({number_pat})\s*([\+\-\*/])\s*({number_pat})\s*=\s*({number_pat})")
        left_re = re.compile(r"\(\s*left\s*:\s*([^\)]+)\)", re.IGNORECASE)

        # Helper to reconstruct (left: ...) using current numbers and the equation
        def _format_num(x: float) -> str:
            # Normalize numeric string: integer if close to int; else trim trailing zeros
            xi = int(round(x))
            if abs(x - xi) < 1e-9:
                return str(xi)
            s = f"{x}"
            if "." in s:
                s = s.rstrip("0").rstrip(".")
            return s

        def reconstruct_c_and_left(a_str: str, op: str, b_str: str, c_str: str) -> tuple[str, str]:
            """Validate operands against current numbers and build canonical left.

            Returns (c_formatted, canonical_action) where canonical_action is
            "A op B = C (left: ...)", or (None, None) if validation fails.
            """
            EPS = 1e-6
            # Current tokens and a working copy to remove operands from
            cur_tokens = state.current.replace(",", " ").split()
            # Helper to pop one instance of a numeric value from tokens
            def pop_value(tokens: list[str], val: float) -> bool:
                for i, t in enumerate(tokens):
                    try:
                        if abs(float(t) - val) < EPS:
                            tokens.pop(i)
                            return True
                    except Exception:
                        continue
                return False

            try:
                a_val = float(a_str)
                b_val = float(b_str)
            except Exception:
                return None, None

            # Validate membership of operands
            tokens_after = cur_tokens.copy()
            if not pop_value(tokens_after, a_val):
                return None, None
            if not pop_value(tokens_after, b_val):
                return None, None

            # Compute C from operation and compare with provided c_str
            try:
                if op == '+':
                    c_calc = a_val + b_val
                elif op == '-':
                    c_calc = a_val - b_val
                elif op == '*':
                    c_calc = a_val * b_val
                elif op == '/':
                    if abs(b_val) < EPS:
                        return None, None
                    c_calc = a_val / b_val
                else:
                    return None, None
            except Exception:
                return None, None

            # Prefer the model's printed C if close; otherwise use computed
            try:
                c_provided = float(c_str)
                if abs(c_calc - c_provided) <= 1e-6:
                    c_val = c_provided
                else:
                    c_val = c_calc
            except Exception:
                c_val = c_calc

            # Build left list: result first, then remaining tokens (in their original string forms)
            left_tokens = [_format_num(c_val)] + tokens_after
            # Validate left size == k-1
            if len(left_tokens) != max(1, len(cur_tokens) - 1):
                return None, None

            c_fmt = _format_num(c_val)
            canonical = f"{_format_num(float(a_str))} {op} {_format_num(float(b_str))} = {c_fmt} (left: {' '.join(left_tokens)})"
            return c_fmt, canonical

        seen = set()
        for ln in lines:
            # Skip any premature Answer lines here; answer is handled elsewhere
            if ln.lower().startswith('answer:'):
                continue
            m = eq_re.search(ln)
            if not m:
                continue
            a, op, b, c = m.groups()
            # Validate and reconstruct left deterministically from the current numbers
            c_fmt, canonical = reconstruct_c_and_left(a, op, b, c)
            if canonical is None:
                continue
            canonical = canonical.replace('  ', ' ').strip()
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
            # Determine how many numbers must remain after one operation
            cur_nums = state.current.replace(',', ' ').split()
            left_count = max(1, len(cur_nums) - 1)
            # Build a tiny example consistent with the current state
            example_line = ''
            try:
                if len(cur_nums) >= 2:
                    a, b = int(cur_nums[0]), int(cur_nums[1])
                    c = a + b
                    remaining = cur_nums[2:]
                    left_list = ' '.join([str(c)] + remaining)
                    example_line = f"Example: {a} + {b} = {c} (left: {left_list})\n"
            except Exception:
                pass
            constraint_tail = (
                f"Output exactly {self.n_actions} lines.\n"
                f"Each line must be of the form: A op B = C (left: exactly {left_count} numbers separated by spaces).\n"
                f"Use only the numbers from: {state.current}.\n"
                + (example_line if example_line else '') +
                "Do not add any analysis, numbering, tags, or extra text.\n"
                "Do not include <think>. Start immediately with the first line.\n"
                "The first character of your output must be a digit.\n"
            )
            prompt = base_prompt + constraint_tail
            print(f'DEBUG: Prompt sent to model: {repr(prompt)}')
            output = self._gen(
                [prompt],
                num_return_sequences=1,
                do_sample=False,
                max_new_tokens=512,
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
