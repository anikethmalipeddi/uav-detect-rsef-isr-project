from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file())


def dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def verify_tsms(
    root: Path,
    catalog_csv: Path,
    min_total_gb: float,
    min_extracted_files_per_archive: int,
) -> tuple[bool, dict]:
    if not root.exists():
        raise FileNotFoundError(f"TSMS root not found: {root}")
    if not catalog_csv.exists():
        raise FileNotFoundError(f"TSMS catalog not found: {catalog_csv}")

    catalog = pd.read_csv(catalog_csv)
    required_cols = {"id", "name", "size_bytes", "folder"}
    missing = required_cols - set(catalog.columns)
    if missing:
        raise ValueError(f"Catalog missing required columns: {sorted(missing)}")

    rows: list[dict] = []
    for _, r in catalog.iterrows():
        file_id = int(r["id"])
        name = str(r["name"])
        size_bytes = int(r["size_bytes"])
        folder = "" if pd.isna(r["folder"]) else str(r["folder"])
        sub = folder.replace("/", "\\")
        dst_dir = root / sub
        base = Path(name).stem
        ext = Path(name).suffix
        is_archive = ext.lower() == ".7z"
        folder_low = folder.replace("\\", "/").lower()
        is_dataset_entry = "/dataset" in folder_low
        # "Required" means entries needed for full TSMS dataset validation.
        required_entry = is_dataset_entry and is_archive
        archive_name = f"{base}_{file_id}{ext}"
        archive_path = dst_dir / archive_name
        extract_dir = dst_dir / f"{base}_{file_id}"

        archive_exists = archive_path.exists()
        archive_size_ok = archive_exists and archive_path.stat().st_size == size_bytes
        extracted_file_count = count_files(extract_dir) if is_archive else 0
        extracted_ok = (
            extract_dir.exists() and extracted_file_count >= int(min_extracted_files_per_archive)
        ) if is_archive else archive_size_ok

        rows.append(
            {
                "id": file_id,
                "name": name,
                "folder": folder,
                "is_archive": is_archive,
                "is_dataset_entry": is_dataset_entry,
                "required_entry": required_entry,
                "archive_path": str(archive_path),
                "extract_dir": str(extract_dir),
                "archive_exists": archive_exists,
                "archive_size_ok": archive_size_ok,
                "expected_size_bytes": size_bytes,
                "extracted_ok": extracted_ok,
                "extracted_file_count": extracted_file_count,
            }
        )

    df = pd.DataFrame(rows)
    required_df = df[df["required_entry"]].copy()
    optional_df = df[~df["required_entry"]].copy()
    total_size_gb = dir_size_bytes(root) / (1024**3)
    extracted_all = bool(required_df["extracted_ok"].all())
    bad_archives = required_df[(required_df["archive_exists"]) & (~required_df["archive_size_ok"])]
    min_size_ok = total_size_gb >= float(min_total_gb)

    # Expected TSMS structure markers for full-dataset state validation.
    required_dirs = [
        root / "TSMS-Drone" / "Dataset",
        root / "TSMS-Drone" / "Dataset" / "FMCW Radar",
    ]
    required_dirs_ok = all(p.exists() and p.is_dir() for p in required_dirs)
    complete = extracted_all and bad_archives.empty and min_size_ok and required_dirs_ok

    state = {
        "root": str(root),
        "catalog_csv": str(catalog_csv),
        "expected_archives": int(len(df)),
        "expected_archive_entries": int(df["is_archive"].sum()),
        "expected_nonarchive_entries": int((~df["is_archive"]).sum()),
        "required_entries": int(len(required_df)),
        "required_extracted_ok_count": int(required_df["extracted_ok"].sum()),
        "optional_entries": int(len(optional_df)),
        "optional_extracted_ok_count": int(optional_df["extracted_ok"].sum()),
        "extracted_ok_count": int(df["extracted_ok"].sum()),
        "archive_exists_count": int(df["archive_exists"].sum()),
        "archive_size_ok_count": int(df["archive_size_ok"].sum()),
        "bad_archive_count": int(len(bad_archives)),
        "min_extracted_files_per_archive": int(min_extracted_files_per_archive),
        "total_size_gb": round(float(total_size_gb), 3),
        "min_total_gb": float(min_total_gb),
        "min_size_ok": bool(min_size_ok),
        "required_dirs": [str(x) for x in required_dirs],
        "required_dirs_ok": bool(required_dirs_ok),
        "complete": bool(complete),
        "missing_extracted": required_df.loc[
            ~required_df["extracted_ok"], ["id", "name", "is_archive", "archive_path", "extract_dir"]
        ]
        .head(30)
        .to_dict(orient="records"),
        "optional_missing": optional_df.loc[
            ~optional_df["extracted_ok"], ["id", "name", "is_archive", "archive_path", "extract_dir"]
        ]
        .head(30)
        .to_dict(orient="records"),
        "bad_archives": bad_archives.loc[:, ["id", "name", "archive_path"]]
        .head(30)
        .to_dict(orient="records"),
    }
    return complete, state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/raw/tsms_drone"))
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--min-total-gb", type=float, default=150.0)
    parser.add_argument("--min-extracted-files-per-archive", type=int, default=1000)
    parser.add_argument("--out-txt", type=Path, default=Path("results/tsms_state.txt"))
    parser.add_argument("--out-json", type=Path, default=Path("results/tsms_state.json"))
    parser.add_argument("--complete-flag", type=Path, default=Path("results/tsms_download_complete.flag"))
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    catalog = args.catalog if args.catalog is not None else (args.root / "figshare_files.csv")
    complete, state = verify_tsms(
        args.root,
        catalog,
        min_total_gb=args.min_total_gb,
        min_extracted_files_per_archive=args.min_extracted_files_per_archive,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_txt.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(state, indent=2), encoding="utf-8")

    lines = [
        "TSMS Verification",
        f"root={state['root']}",
        f"catalog={state['catalog_csv']}",
        f"expected_archives={state['expected_archives']}",
        f"required_entries={state['required_entries']}",
        f"required_extracted_ok_count={state['required_extracted_ok_count']}",
        f"optional_entries={state['optional_entries']}",
        f"optional_extracted_ok_count={state['optional_extracted_ok_count']}",
        f"extracted_ok_count={state['extracted_ok_count']}",
        f"archive_exists_count={state['archive_exists_count']}",
        f"archive_size_ok_count={state['archive_size_ok_count']}",
        f"bad_archive_count={state['bad_archive_count']}",
        f"min_extracted_files_per_archive={state['min_extracted_files_per_archive']}",
        f"total_size_gb={state['total_size_gb']}",
        f"min_total_gb={state['min_total_gb']}",
        f"min_size_ok={state['min_size_ok']}",
        f"required_dirs_ok={state['required_dirs_ok']}",
    ]
    if complete:
        lines.append("TSMS_STATE=COMPLETE")
        lines.append("TSMS_DATASET_COMPLETE=TRUE")
        lines.append("COMPLETE")
    else:
        lines.append("TSMS_STATE=INCOMPLETE")
        lines.append("TSMS_DATASET_COMPLETE=FALSE")
        lines.append("INCOMPLETE")
        if state["missing_extracted"]:
            lines.append("Missing extracted entries (first 30):")
            for x in state["missing_extracted"]:
                if bool(x["is_archive"]):
                    lines.append(f"- id={x['id']} name={x['name']} extract_dir={x['extract_dir']}")
                else:
                    lines.append(f"- id={x['id']} name={x['name']} file_path={x['archive_path']}")
        if state["optional_missing"]:
            lines.append("Optional non-dataset missing entries (first 30):")
            for x in state["optional_missing"]:
                if bool(x["is_archive"]):
                    lines.append(f"- id={x['id']} name={x['name']} extract_dir={x['extract_dir']}")
                else:
                    lines.append(f"- id={x['id']} name={x['name']} file_path={x['archive_path']}")
        if state["bad_archives"]:
            lines.append("Bad archives (first 30):")
            for x in state["bad_archives"]:
                lines.append(f"- id={x['id']} name={x['name']} archive_path={x['archive_path']}")

    args.out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out_txt.read_text(encoding="utf-8"))

    args.complete_flag.parent.mkdir(parents=True, exist_ok=True)
    if complete:
        args.complete_flag.write_text("TSMS_DOWNLOAD_COMPLETE=TRUE\n", encoding="utf-8")
    elif args.complete_flag.exists():
        args.complete_flag.unlink()

    if not complete and not args.allow_incomplete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
