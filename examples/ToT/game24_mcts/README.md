# MCTS over Tree of Thoughts for Game24

This implementation combines **Monte Carlo Tree Search (MCTS)** with **Tree of Thoughts (ToT)** methodology for Game24 problems, with support for **SGLang backend** for optimized inference.

## What This Achieves
✅ **MCTS + ToT**: Adaptive reasoning exploration instead of fixed beam search  
✅ **SGLang Backend**: 100x faster inference with optimized server  
✅ **Pluggable Design**: Easy switching between algorithms and backends  

## Key Differences from Original ToT
- **Algorithm**: MCTS instead of BeamSearch
- **Exploration**: UCB1-based adaptive exploration vs fixed beam width  
- **Search Strategy**: Iterative deepening with uncertainty-aware selection
- **Backend**: SGLang server support for speed optimization

## Setup

### For SGLang Backend (Recommended)
1. Start SGLang server:
```bash
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --port 30001
```

2. Run MCTS-ToT with SGLang:
```bash
python examples/ToT/game24_mcts/inference.py --base_lm sglang --sglang_model meta-llama/Llama-3.1-8B-Instruct --sglang_url http://127.0.0.1:30001
```

### Other Backends

HuggingFace (local):
```bash
python examples/ToT/game24_mcts/inference.py --base_lm hf --hf_path meta-llama/Llama-3.1-8B-Instruct --batch_size 8
```

Llama3 (local):
```bash
python examples/ToT/game24_mcts/inference.py --base_lm llama-3 --llama_3_ckpts $LLAMA3_CKPTS --llama_size "8B-Instruct" --batch_size 8
```

Llama2 (local):
```bash
python examples/ToT/game24_mcts/inference.py --base_lm llama-2 --llama_2_ckpts your/path/to/llama --llama_size "7B" --batch_size 8
```