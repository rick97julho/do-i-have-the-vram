import json
import struct
import os
import hashlib
from pathlib import Path
import requests
from huggingface_hub import list_repo_files, hf_hub_url, get_token, model_info
from typing import List, Dict, Any, Optional

CACHE_DIR = Path.home() / ".cache" / "do-i-have-the-vram"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _get_auth_headers(token: Optional[str] = None) -> Dict[str, str]:
    token = token or get_token()
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}

def get_model_sha(repo_id: str, revision: str = "main", token: Optional[str] = None) -> Optional[str]:
    """
    Get the commit SHA for the given model revision.
    """
    try:
        info = model_info(repo_id, revision=revision, token=token)
        return info.sha
    except Exception as e:
        if "401" in str(e):
             print(f"Error: Unauthorized to access {repo_id}. Please log in or provide a token.")
        else:
             print(f"Error fetching model info for {repo_id}: {e}")
        return None

def get_safetensors_files(repo_id: str, revision: str = "main", token: Optional[str] = None) -> List[str]:
    """
    List all .safetensors files in the repository.
    """
    try:
        files = list_repo_files(repo_id, revision=revision, token=token)
        return [f for f in files if f.endswith(".safetensors")]
    except Exception as e:
        if "401" in str(e):
            print(f"Error: Unauthorized to access {repo_id}.")
        else:
            print(f"Error listing files for {repo_id}: {e}")
        return []

def _get_cache_path(url: str) -> Path:
    # Use hash of URL for cache filename to handle special chars and length
    url_hash = hashlib.sha256(url.encode()).hexdigest()
    return CACHE_DIR / f"{url_hash}.json"

def fetch_safetensors_header(url: str, token: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch and parse the header of a .safetensors file using range requests.
    Caches the header locally.
    """
    cache_path = _get_cache_path(url)
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass # Invalid cache, ignore

    headers = _get_auth_headers(token)
    
    try:
        # Read first 8 bytes to get header size (uint64 little-endian)
        # We need to merge Range header with Auth header
        req_headers = headers.copy()
        req_headers["Range"] = "bytes=0-7"
        
        r = requests.get(url, headers=req_headers)
        r.raise_for_status()
        header_len = struct.unpack("<Q", r.content)[0]

        # Read header JSON
        req_headers["Range"] = f"bytes=8-{8 + header_len - 1}"
        r = requests.get(url, headers=req_headers)
        r.raise_for_status()
        
        header = json.loads(r.content)
        
        # Save to cache
        with open(cache_path, "w") as f:
            json.dump(header, f)
            
        return header
    except Exception as e:
        if "401" in str(e):
            print(f"Error: Unauthorized to fetch header from {url}.")
        else:
            print(f"Error fetching header from {url}: {e}")
        return {}

def fetch_config(repo_id: str, revision: str = "main", token: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Fetch the config.json file from the repository.
    """
    url = hf_hub_url(repo_id, "config.json", revision=revision)
    headers = _get_auth_headers(token)
    
    try:
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        if "401" in str(e):
             print(f"Error: Unauthorized to fetch config for {repo_id}.")
        else:
            print(f"Error fetching config for {repo_id}: {e}")
        return None

def get_file_url(repo_id: str, filename: str, revision: str = "main") -> str:
    """
    Get the downloadable URL for a file in the repo.
    """
    return hf_hub_url(repo_id, filename, revision=revision)
