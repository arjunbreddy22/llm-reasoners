from typing import Literal, Optional

from reasoners import LanguageModel
from prompts.game24 import standard_prompt
from datetime import datetime
import os
import sys
import utils
from tqdm import tqdm
import numpy as np
import time
import re


def cot_game24(base_model: LanguageModel, disable_log: bool = False, resume=0, 
               results_model_label: Optional[str] = None, results_dir: Optional[str] = None, **kwargs):
    if not disable_log:
        log_dir = f'logs/game24_cot/{datetime.now().strftime("%m%d%Y-%H%M%S")}'
        os.makedirs(log_dir)
        os.makedirs(os.path.join(log_dir, 'algo_output'), exist_ok=True)
        with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
            print(sys.argv, file=f)
    # test from 900-910 for 10-problem test (change back to 900:1000 for full test)
    dataset = utils.read_data(file='./examples/ToT/game24/data/24.csv')[900:910]
    correct_count = 0
    latencies_ms = []
    for i, example in enumerate(tqdm(dataset, total=len(dataset), initial=0, desc='game24', disable=disable_log)):
        lm_input = standard_prompt.format(input=example)
        
        # Debug: Check what we're actually sending to the model
        print(f"DEBUG: example = {repr(example)}")
        print(f"DEBUG: lm_input = {repr(lm_input)}")
        
        # Time the single LLM call for sequential decoding
        # Avoid stopping on a single newline to prevent truncation at "<think>\n".
        # Also avoid adding extra CONTINUE templates; keep the prompt minimal and parse the result.
        start_time = time.perf_counter()
        raw = base_model.generate([lm_input], temperature=0.0, do_sample=False, max_new_tokens=256).text[0]
        end_time = time.perf_counter()
        # Post-process to remove <think> blocks and extract the first equation line
        text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)
        # Also drop unmatched think tags if the block wasn't closed
        text = text.replace("<think>", "").replace("</think>", "")
        lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
        # Prefer a line that contains an equation and optionally ends with = 24
        candidates = []
        for ln in lines:
            s = ln
            if s.lower().startswith('answer:'):
                s = s[len('answer:'):].strip()
            if '=' in s:
                candidates.append(s)
        output = None
        # Pick the first candidate explicitly asserting 24 on RHS
        for c in candidates:
            if re.search(r"=\s*24\b", c):
                output = c
                break
        # Fallback: pick the first equation line if none explicitly say 24
        if output is None and candidates:
            output = candidates[0]
        if output is None:
            # Second attempt: force a single-line answer without analysis
            force_tail = (
                "\nOnly output one line in the exact format: (EXPRESSION) = 24\n"
                "Do not include any other text. No <think>.\n"
                "Answer: "
            )
            raw2 = base_model.generate([lm_input + force_tail], temperature=0.0, do_sample=False, max_new_tokens=128).text[0]
            text2 = re.sub(r"<think>.*?</think>", "", raw2, flags=re.DOTALL | re.IGNORECASE)
            text2 = text2.replace("<think>", "").replace("</think>", "")
            line2 = text2.strip().split('\n')[0]
            # Normalize to include 'Answer:' prefix once
            if not line2.lower().startswith('answer:'):
                output = f"Answer: {line2.strip()}"
            else:
                output = line2.strip()
            # Final guard: if still no '=', fall back to first non-empty line of original
            if '=' not in output and lines:
                output = lines[0]
        print(f"DEBUG: raw = {repr(raw)}")
        if output is None:
            print(f"DEBUG: second raw = {repr(raw2)}")
        print(f"DEBUG: parsed output = {repr(output)}")
        latency_ms = (end_time - start_time) * 1000.0
        latencies_ms.append(latency_ms)
        
        correct = utils.test_output(example, output)
        correct_count += correct
        accuracy = correct_count / (i + 1)
        log_str = f'Case #{resume + i + 1}: {correct=}, {output=} ; {accuracy=:.3f} ({correct_count}/{i + 1}); latency_ms={latency_ms:.1f}'
        if not disable_log:
            tqdm.write(log_str)
            with open(os.path.join(log_dir, 'result.log'), 'a') as f:
                print(log_str, file=f)
    
    # Persist per-problem latencies and total accuracy to a results file
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    if results_dir is None:
        results_dir = os.path.join('results', 'game24_cot')
    os.makedirs(results_dir, exist_ok=True)
    def _sanitize(s: str) -> str:
        try:
            return re.sub(r'[^A-Za-z0-9_.\-]+', '_', s)
        except Exception:
            return 'model'
    model_tag = _sanitize(results_model_label or 'model')
    results_path = os.path.join(results_dir, f'{timestamp}_{model_tag}.json')
    summary = {
        'timestamp': timestamp,
        'model': results_model_label,
        'n_examples': len(dataset),
        'latencies_ms': latencies_ms,
        'mean_latency_ms': float(np.nan if len([x for x in latencies_ms if x is not None]) == 0 else float(np.mean([x for x in latencies_ms if x is not None]))),
        'total_accuracy': float(correct_count / max(1, len(dataset))),
    }
    with open(results_path, 'w') as f:
        import json as _json
        _json.dump(summary, f, indent=2)
    print(f'CoT results saved to {results_path}')

if __name__ == '__main__':
    import os
    import sys
    import json
    import warnings
    import fire
    import random

    llama_ckpts = os.environ.get("LLAMA_CKPTS", None)
    llama_2_ckpts = os.environ.get("LLAMA_2_CKPTS", None)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank != 0:
        sys.stdout = open(os.devnull, 'w')
        warnings.filterwarnings('ignore')

    def main(base_lm: Literal['llama', 'llama.cpp', 'llama-2', 'hf', 'exllama', 'sglang'] = 'sglang',
             llama_ckpts: str = llama_ckpts,
             llama_2_ckpts: str = llama_2_ckpts,
             llama_size: str = '13B',
             llama_cpp_path: str = None,
             llama_cpp_n_batch: int = 512,
             hf_path: str = 'meta-llama/Llama-2-13b-hf',
             hf_peft_path: Optional[str] = None,
             hf_quantized: Optional[Literal['awq', 'int8', 'fp4', 'nf4']] = None,
             hf_load_awq_path: Optional[str] = None,
             exllama_model_dir: str = 'WizardMath-13B-V1.0-GPTQ',
             exllama_lora_dir: Optional[str] = None,
             exllama_mem_map: Optional[str] = None,
             sglang_url: str = 'http://127.0.0.1:30001',
             sglang_model: str = 'Qwen/Qwen2.5-7B-Instruct',
             batch_size: int = 1,
             openai_mode: str = 'gpt-4-1106-preview',
             disable_log: bool = False,
             disable_tqdm: bool = False,
             **kwargs):
        if base_lm in ['llama', 'llama2']:
            import torch
            import torch.backends.cudnn
            np.random.seed(0)
            random.seed(0)
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            torch.backends.cudnn.deterministic = True

        if base_lm == 'llama':
            from reasoners.lm import LlamaModel
            base_model = LlamaModel(llama_ckpts, llama_size, max_batch_size=batch_size)
        elif base_lm == 'llama.cpp':
            from reasoners.lm import LlamaCppModel
            base_model = LlamaCppModel(llama_cpp_path, n_batch=llama_cpp_n_batch)
        elif base_lm == 'llama-2':
            from reasoners.lm import Llama2Model
            base_model = Llama2Model(llama_2_ckpts, llama_size, max_batch_size=batch_size)
        elif base_lm == 'hf':
            from reasoners.lm import HFModel
            base_model = HFModel(hf_path, hf_path, max_batch_size=batch_size, max_new_tokens=512,
                                 peft_pth=hf_peft_path, quantized=hf_quantized, load_awq_pth=hf_load_awq_path)
        elif base_lm == 'exllama':
            from reasoners.lm import ExLlamaModel
            base_model = ExLlamaModel(exllama_model_dir, exllama_lora_dir, mem_map=exllama_mem_map,
                                      max_batch_size=batch_size, max_new_tokens=512, max_seq_length=2048)
        elif base_lm == 'openai':
            from reasoners.lm import OpenAIModel
            base_model = OpenAIModel(openai_mode)
        elif base_lm == 'gemini':
            from reasoners.lm import BardCompletionModel
            base_model = BardCompletionModel('gemini-pro')
        elif base_lm == 'claude':
            from reasoners.lm import ClaudeModel
            base_model = ClaudeModel('claude-3-opus-20240229')
        elif base_lm == 'sglang':
            import os
            from reasoners.lm import SGLangModel
            os.environ["SGLANG_API_URL"] = sglang_url
            base_model = SGLangModel(sglang_model, max_new_tokens=1024, is_instruct_model=True)
        else:
            assert False, f'cannot resolve {base_lm=}'
        # Determine a model label for results naming
        def _model_label():
            try:
                if base_lm == 'sglang':
                    return sglang_model
                if base_lm == 'hf':
                    return hf_path
                if base_lm == 'llama-2':
                    return f"llama-2_{llama_size}"
                if base_lm == 'llama':
                    return f"llama_{llama_size}"
                if base_lm == 'exllama':
                    return exllama_model_dir
                if base_lm == 'llama.cpp':
                    return 'llama.cpp'
            except Exception:
                pass
            return base_lm
        
        cot_game24(base_model=base_model, disable_log=disable_log or local_rank > 0, 
                  results_model_label=_model_label(), **kwargs)


    fire.Fire(main)
