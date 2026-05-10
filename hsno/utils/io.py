from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def append_jsonl(path: str | Path, rows: Iterable[dict], overwrite: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "a"
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    append_jsonl(path, rows, overwrite=True)


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv_jsonl(rows: list[dict], csv_path: str | Path, jsonl_path: str | Path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    write_jsonl(jsonl_path, rows)
    return df
