import copy
import dataclasses
import re
from typing import Optional
from reasoners import WorldModel, LanguageModel


@dataclasses.dataclass
class Game24State:
    input: str
    current: str
    history: list[str]
    output: Optional[str] = None

    def __str__(self):
        if self.output is None:
            return f'Game24State({repr(self.current)})'
        else:
            return f'Game24State({repr(self.current)}, output={repr(self.output)})'


Game24Action = str


class Game24WorldModel(WorldModel):
    def __init__(self,
                 base_model: LanguageModel,
                 prompt: dict,
                 n_confidence=8,
                 batch_size=2, ) -> None:
        super().__init__()
        self.base_model = base_model
        self.prompt = prompt
        self.batch_size = batch_size
        self.n_confidence = n_confidence

    def init_state(self) -> Game24State:
        return Game24State(self.example, self.example, [])

    def step(self, state: Game24State, action: Game24Action) -> tuple[Game24State, dict]:
        next_state = copy.deepcopy(state)
        if 'Answer' in action:
            match = re.match(r'Answer: (.*)', action)
            next_state.output = match[1] if match is not None else ''
        else:
            match = re.match(r'.*\(left: (.*)\)', action)
            if match is None:
                # Attempt to reconstruct a correct left-list using utilities
                try:
                    from utils import correct_left_numbers
                    reconstructed = correct_left_numbers(state.input, '\n'.join(state.history), action)
                    match = re.match(r'.*\(left: (.*)\)', reconstructed)
                    if match is not None:
                        action = reconstructed
                except Exception:
                    pass
            if match is not None:
                # Normalize format: remove commas to match prompt examples
                next_state.current = match[1].replace(',', '').replace('  ', ' ').strip()
                next_state.history.append(action)
            else:
                # If still not parsable, keep state unchanged and do not append invalid action
                print(f'DEBUG: step could not parse or reconstruct left-list for action: {repr(action)}')
        print(f'DEBUG: Stepping {state} with {action=} to {next_state}')
        return next_state, {'next_state': next_state}

    def is_terminal(self, state: Game24State) -> bool:
        is_term = state.output is not None
        print(f'DEBUG: is_terminal check - state.current={repr(state.current)}, state.output={repr(state.output)}, is_terminal={is_term}')
        if is_term:
            print(f'DEBUG: Terminal state reached! Output: {repr(state.output)}')
        return is_term
