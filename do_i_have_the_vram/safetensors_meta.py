from typing import Dict, Any, Tuple, Union

# Size in bytes for each dtype
DTYPE_SIZES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    # Common aliases/variations found in some tools
    "float64": 8,
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "int8": 1,
    "uint8": 1,
    "bool": 1,
    # Quantized types
    "int4": 0.5,
    "fp8": 1,
    "float8": 1,
    "4bit": 0.5,
    "8bit": 1,
}

def get_dtype_size(dtype_str: str) -> float:
    """
    Get size in bytes for a given dtype string.
    """
    # Normalize case usually not needed for standard safetensors but good for robustness
    size = DTYPE_SIZES.get(dtype_str)
    if size is None:
        # Fallback: sometimes dtypes might be lowercase in some contexts
        size = DTYPE_SIZES.get(dtype_str.lower())
    
    if size is None:
        # If unknown, warn and assume 2 bytes (F16) but print warning.
        print(f"Warning: Unknown dtype '{dtype_str}', assuming 2 bytes.")
        return 2.0
    return size

def calculate_tensor_bytes(shape: list, dtype: str) -> float:
    """
    Calculate the number of bytes a tensor occupies.
    """
    if not shape:
        return 0.0
    
    num_elements = 1
    for dim in shape:
        num_elements *= dim
        
    return num_elements * get_dtype_size(dtype)

def parse_header(header: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    Parse a safetensors header and calculate total size of tensors.
    Returns:
        total_bytes: Total size in bytes of all tensors in this file.
        tensors_info: Dictionary mapping tensor names to their calculated size.
    """
    total_bytes = 0.0
    tensors_info = {}
    
    # The header contains a "__metadata__" key which is not a tensor
    for key, value in header.items():
        if key == "__metadata__":
            continue
            
        shape = value.get("shape")
        dtype = value.get("dtype")
        
        if shape is not None and dtype is not None:
            size = calculate_tensor_bytes(shape, dtype)
            total_bytes += size
            tensors_info[key] = {
                "shape": shape,
                "dtype": dtype,
                "size": size
            }
            
    return total_bytes, tensors_info
