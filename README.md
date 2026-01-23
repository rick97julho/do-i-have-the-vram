# Do I Have The VRAM?

![PyPI version](https://img.shields.io/pypi/v/do-i-have-the-vram)

**"Can I run this model?"** — The question we ask every time a new LLM drops.

`do-i-have-the-vram` is a lightweight CLI tool that estimates exactly how much VRAM you need to run a Hugging Face model **without downloading it first**.

It works by fetching only the metadata (headers) of `.safetensors` files using HTTP Range requests. This allows it to calculate precise memory usage for weights, KV cache, and activations in seconds.

## Features

- **Instant Estimation**: No 50GB downloads. Fetches KB of metadata.
- **Accurate**: Uses exact tensor shapes from the model files.
- **Context Aware**: Calculates KV cache and activation memory based on your batch size and sequence length.
- **Quantization Support**: Estimates VRAM for `int8`, `int4`, `float16`, etc.
- **Visual Breakdown**: See where the memory goes (Weights vs KV Cache vs Activations).
- **Secure**: Supports gated/private models via Hugging Face tokens.

## Installation

### From PyPI (Recommended)
```bash
pip install do-i-have-the-vram
```

### From Source
```bash
git clone https://github.com/yourusername/do-i-have-the-vram.git
cd do-i-have-the-vram
pip install -e .
```

## Usage

Basic usage requires just the model ID:

```bash
do-i-have-the-vram meta-llama/Llama-2-7b-hf
```

### Common Options

| Flag | Description | Default |
|------|-------------|---------|
| `--dtype` | Inference data type (`float16`, `int8`, `int4`, `fp32`) | `float16` |
| `--batch` | Batch size | `1` |
| `--seq-len` | Sequence length (Context window) | `4096` |
| `--no-kv` | Disable KV cache estimation (e.g. for training/no-cache inference) | `False` |
| `--json` | Output results in raw JSON (useful for scripts) | `False` |
| `--token` | Hugging Face token (for gated models like Llama 3) | `None` |

### Examples

**1. Can I run Llama-2-70B on my 24GB card with 4-bit quantization?**
```bash
do-i-have-the-vram meta-llama/Llama-2-70b-hf --dtype int4
```

**2. How much VRAM for a long-context window?**
```bash
do-i-have-the-vram mistralai/Mistral-7B-v0.1 --seq-len 32000
```

**3. Check a gated model (requires login or token)**
```bash
# If you are already logged in via `huggingface-cli login`:
do-i-have-the-vram meta-llama/Meta-Llama-3-8B

# Or pass token explicitly:
do-i-have-the-vram meta-llama/Meta-Llama-3-8B --token hf_abc123...
```

## How it Works

1. **Resolution**: The tool resolves the model's commit SHA to ensure consistency.
2. **Config**: Fetches `config.json` to understand the architecture (layers, heads, dimensions).
3. **Headers**: Lists `.safetensors` files and performs **HTTP Range requests** to download *only* the first few kilobytes of each file.
4. **Parsing**: Reads the JSON header inside the safetensors file to get the exact shape of every tensor (weights).
5. **Calculation**: 
    - **Weights**: Sum of (params × dtype_size).
    - **KV Cache**: `2 × layers × heads × head_dim × seq_len × dtype_size`.
    - **Activations**: Approximate overhead based on batch size and hidden size.

## Contributing

Pull requests are welcome! 

1. Fork the repo.
2. Create a new branch.
3. Install dev dependencies: `pip install -e .`
4. Make your changes.
5. Submit a PR.

## License

MIT
