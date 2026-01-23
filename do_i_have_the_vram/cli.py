import argparse
import sys
import json
from typing import Union

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track, Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich import print as rprint

from do_i_have_the_vram.hub_client import get_safetensors_files, fetch_safetensors_header, fetch_config, get_file_url, get_model_sha
from do_i_have_the_vram.safetensors_meta import get_dtype_size
from do_i_have_the_vram.memory_model import estimate_vram_usage

console = Console()

def format_bytes(size: Union[int, float]) -> str:
    power = 2**30 # 1024**3 for GB
    n = size / power
    return f"{n:.2f} GB"

def categorize_tensor(name: str) -> str:
    name = name.lower()
    if any(k in name for k in ["attn", "attention", "q_proj", "k_proj", "v_proj", "o_proj", "c_attn", "c_proj"]):
        return "attention"
    if any(k in name for k in ["mlp", "feed_forward", "gate_proj", "up_proj", "down_proj", "fc1", "fc2"]):
        return "mlp"
    if any(k in name for k in ["embed", "wte", "wpe"]):
        return "embedding"
    if any(k in name for k in ["ln", "layer_norm", "norm"]):
        return "norm"
    return "other"

def main():
    parser = argparse.ArgumentParser(description="Estimate VRAM usage for Hugging Face models without downloading them.")
    parser.add_argument("model_id", type=str, help="Hugging Face model ID (e.g. meta-llama/Llama-2-7b-hf)")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=4096, help="Sequence length")
    parser.add_argument("--dtype", type=str, default="float16", help="Data type for inference (float16, float32, int8, int4, etc.)")
    parser.add_argument("--no-kv", action="store_true", help="Disable KV cache estimation")
    parser.add_argument("--token", type=str, help="Hugging Face token (optional, for gated models)")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    args = parser.parse_args()

    if args.json:
        run_json_mode(args)
    else:
        run_interactive_mode(args)

def run_json_mode(args):
    # Quiet mode for JSON output
    revision = "main"
    sha = get_model_sha(args.model_id, revision=revision, token=args.token)
    if sha:
        revision = sha

    config = fetch_config(args.model_id, revision=revision, token=args.token)
    if not config:
        sys.exit(1)

    files = get_safetensors_files(args.model_id, revision=revision, token=args.token)
    if not files:
        sys.exit(1)

    total_weight_bytes = 0.0
    target_dtype_size = get_dtype_size(args.dtype)
    
    breakdown = {
        "attention": 0.0,
        "mlp": 0.0,
        "embedding": 0.0,
        "norm": 0.0,
        "other": 0.0
    }

    for file in files:
        url = get_file_url(args.model_id, file, revision=revision)
        header = fetch_safetensors_header(url, token=args.token)
        if not header:
            continue
            
        for key, value in header.items():
            if key == "__metadata__":
                continue
            
            shape = value.get("shape")
            if shape:
                num_elements = 1
                for dim in shape:
                    num_elements *= dim
                
                size = num_elements * target_dtype_size
                total_weight_bytes += size
                
                category = categorize_tensor(key)
                breakdown[category] += size

    estimates = estimate_vram_usage(
        weight_bytes=total_weight_bytes,
        config=config,
        batch_size=args.batch,
        seq_len=args.seq_len,
        dtype_size=target_dtype_size,
        enable_kv_cache=not args.no_kv
    )

    output = {
        "model_id": args.model_id,
        "revision": revision,
        "inference_dtype": args.dtype,
        "bytes_per_param": target_dtype_size,
        "memory_breakdown_gb": {
            "weights": estimates['weights'] / 2**30,
            "kv_cache": estimates['kv_cache'] / 2**30,
            "activations": estimates['activations'] / 2**30,
            "total": estimates['total'] / 2**30
        },
        "weight_breakdown_gb": {k: v / 2**30 for k, v in breakdown.items()}
    }
    print(json.dumps(output, indent=2))

def run_interactive_mode(args):
    console.print(f"[bold blue]Do I Have The VRAM?[/bold blue] - Estimating for [bold green]{args.model_id}[/bold green]")
    
    with console.status("[bold green]Resolving model revision...[/bold green]", spinner="dots"):
        revision = "main"
        sha = get_model_sha(args.model_id, revision=revision, token=args.token)
        if sha:
            revision = sha
            console.print(f"✔ Resolved revision: [cyan]{sha[:8]}[/cyan]")
        else:
            console.print("⚠ Could not resolve SHA, using main branch.")

    with console.status("[bold green]Fetching config...[/bold green]", spinner="dots"):
        config = fetch_config(args.model_id, revision=revision, token=args.token)
        if not config:
            console.print("[bold red]Error:[/bold red] Could not fetch config.json")
            sys.exit(1)
        console.print("✔ Config fetched")

    with console.status("[bold green]Listing files...[/bold green]", spinner="dots"):
        files = get_safetensors_files(args.model_id, revision=revision, token=args.token)
        if not files:
            console.print("[bold red]Error:[/bold red] No .safetensors files found.")
            sys.exit(1)
        console.print(f"✔ Found [cyan]{len(files)}[/cyan] safetensors files")

    # Processing headers
    total_weight_bytes = 0.0
    target_dtype_size = get_dtype_size(args.dtype)
    
    breakdown = {
        "attention": 0.0,
        "mlp": 0.0,
        "embedding": 0.0,
        "norm": 0.0,
        "other": 0.0
    }

    # Rich progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Parsing headers...[/cyan]", total=len(files))
        
        for file in files:
            url = get_file_url(args.model_id, file, revision=revision)
            header = fetch_safetensors_header(url, token=args.token)
            
            if header:
                for key, value in header.items():
                    if key == "__metadata__":
                        continue
                    
                    shape = value.get("shape")
                    if shape:
                        num_elements = 1
                        for dim in shape:
                            num_elements *= dim
                        
                        size = num_elements * target_dtype_size
                        total_weight_bytes += size
                        
                        category = categorize_tensor(key)
                        breakdown[category] += size
            
            progress.advance(task)

    # Calculate VRAM
    estimates = estimate_vram_usage(
        weight_bytes=total_weight_bytes,
        config=config,
        batch_size=args.batch,
        seq_len=args.seq_len,
        dtype_size=target_dtype_size,
        enable_kv_cache=not args.no_kv
    )

    # Display Results
    
    # 1. Model Info Panel
    info_text = f"""
[bold]Model:[/bold] {args.model_id}
[bold]Revision:[/bold] {revision[:8]}
[bold]Dtype:[/bold] {args.dtype} ({target_dtype_size} bytes/param)
[bold]Context:[/bold] Batch={args.batch}, Seq={args.seq_len}
    """.strip()
    
    console.print()
    console.print(Panel(info_text, title="Model Information", expand=False, border_style="blue"))
    console.print()

    # 2. Memory Table
    table = Table(title="VRAM Usage Estimation", show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan", no_wrap=True)
    table.add_column("Size (GB)", style="green", justify="right")
    table.add_column("Details", style="yellow")

    # Weights
    weight_details = []
    if breakdown['attention'] > 0: weight_details.append(f"Attn: {format_bytes(breakdown['attention'])}")
    if breakdown['mlp'] > 0: weight_details.append(f"MLP: {format_bytes(breakdown['mlp'])}")
    if breakdown['embedding'] > 0: weight_details.append(f"Embed: {format_bytes(breakdown['embedding'])}")
    
    table.add_row(
        "Model Weights", 
        format_bytes(estimates['weights']), 
        ", ".join(weight_details)
    )

    # KV Cache
    table.add_row(
        "KV Cache", 
        format_bytes(estimates['kv_cache']), 
        f"Context Window: {args.seq_len}" if estimates['kv_cache'] > 0 else "Disabled"
    )

    # Activations
    table.add_row(
        "Activations", 
        format_bytes(estimates['activations']), 
        f"Batch Size: {args.batch}"
    )

    # Total
    table.add_section()
    table.add_row(
        "[bold]Total VRAM[/bold]", 
        f"[bold red]{format_bytes(estimates['total'])}[/bold red]", 
        ""
    )

    console.print(table)
    console.print()

if __name__ == "__main__":
    main()
