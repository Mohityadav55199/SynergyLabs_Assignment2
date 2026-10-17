import json
import yaml
from pathlib import Path
from typing import List, Dict, Any, Union

def load_suite_data(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Universally loads test cases from either JSON or YAML files.
    Supports:
    - Raw lists: [ {...}, {...} ]
    - Wrapped dicts: {"test_cases": [...]}, {"cases": [...]}, {"pairs": [...]}, etc.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Suite file not found at: {path}")

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return []

    data = None
    suffix = path.suffix.lower()

    if suffix in [".yaml", ".yml"]:
        data = yaml.safe_load(content)
    else:
        # Default to JSON first, fall back to YAML
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            try:
                data = yaml.safe_load(content)
            except Exception as e:
                raise ValueError(f"Could not parse file as JSON or YAML: {path}. Error: {e}")

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for candidate_key in ["test_cases", "cases", "data", "pairs", "probes", "benchmark"]:
            if candidate_key in data and isinstance(data[candidate_key], list):
                return data[candidate_key]
        # If single object, wrap in list
        return [data]

    raise ValueError(f"Unexpected data format in {path}: expected list or dict, got {type(data)}")
