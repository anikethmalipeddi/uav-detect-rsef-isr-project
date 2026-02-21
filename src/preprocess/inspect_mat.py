from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import scipy.io as sio


def summarize_obj(obj: Any, depth: int = 0, max_depth: int = 4) -> Any:
    if depth > max_depth:
        return "<max_depth>"
    if isinstance(obj, (int, float, str, bool)):
        return obj
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        return {"type": type(obj).__name__, "shape": tuple(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, dict):
        return {str(k): summarize_obj(v, depth + 1, max_depth) for k, v in list(obj.items())[:20]}
    if isinstance(obj, (list, tuple)):
        return [summarize_obj(v, depth + 1, max_depth) for v in obj[:20]]
    if hasattr(obj, "__dict__"):
        out = {"__class__": obj.__class__.__name__}
        for k, v in list(vars(obj).items())[:20]:
            out[k] = summarize_obj(v, depth + 1, max_depth)
        return out
    return {"type": type(obj).__name__}


def inspect_mat(mat_path: Path) -> dict[str, Any]:
    d = sio.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    keys = [k for k in d.keys() if not k.startswith("__")]
    out: dict[str, Any] = {"path": str(mat_path), "keys": keys, "summary": {}}
    for k in keys:
        out["summary"][k] = summarize_obj(d[k], depth=0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("results/mat_inspect.json"))
    args = parser.parse_args()

    result = inspect_mat(args.mat)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
