from typing import Dict, Any, Union

def get_config_value(config: Dict[str, Any], keys: list, default=None):
    """
    Helper to try multiple keys for a config value (e.g. n_head vs num_attention_heads).
    """
    for key in keys:
        if key in config:
            return config[key]
    return default

def calculate_kv_cache_bytes(config: Dict[str, Any], seq_len: int, dtype_size: Union[int, float] = 2) -> float:
    """
    Calculate KV cache size in bytes.
    Formula: 2 * n_layers * n_heads * head_dim * seq_len * dtype_size
    """
    num_hidden_layers = get_config_value(config, ["num_hidden_layers", "n_layer", "n_layers"], 0)
    num_attention_heads = get_config_value(config, ["num_attention_heads", "n_head", "n_heads"], 0)
    hidden_size = get_config_value(config, ["hidden_size", "n_embd", "d_model"], 0)
    
    # If head_dim is not explicitly in config, calculate it
    head_dim = config.get("head_dim")
    if not head_dim and num_attention_heads > 0:
        head_dim = hidden_size // num_attention_heads
    
    if not head_dim:
        # Fallback if we can't determine dimensions
        return 0.0

    return (
        2
        * num_hidden_layers
        * num_attention_heads
        * head_dim
        * seq_len
        * dtype_size
    )

def calculate_activation_bytes(config: Dict[str, Any], batch_size: int, seq_len: int, dtype_size: Union[int, float] = 2) -> float:
    """
    Estimate activation memory.
    Rough approximation: batch * seq_len * hidden_dim * dtype_size * factor (often ~4-6 for transformers)
    This varies significantly by implementation (e.g. flash attention reduces this).
    """
    hidden_size = get_config_value(config, ["hidden_size", "n_embd", "d_model"], 0)
    
    # Factor is a rough heuristic. 
    # For standard attention: Q, K, V, output projection, plus intermediate MLP activations.
    # We'll use a conservative factor.
    factor = 4 
    
    return batch_size * seq_len * hidden_size * dtype_size * factor

def estimate_vram_usage(
    weight_bytes: Union[int, float], 
    config: Dict[str, Any], 
    batch_size: int, 
    seq_len: int, 
    dtype_size: Union[int, float] = 2,
    enable_kv_cache: bool = True
) -> Dict[str, float]:
    """
    Compute total VRAM estimation.
    """
    kv_cache = calculate_kv_cache_bytes(config, seq_len, dtype_size) if enable_kv_cache else 0.0
    activations = calculate_activation_bytes(config, batch_size, seq_len, dtype_size)
    
    total = weight_bytes + kv_cache + activations
    
    return {
        "weights": float(weight_bytes),
        "kv_cache": kv_cache,
        "activations": activations,
        "total": total
    }
