#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

CODE2LABEL = {
    "213": 0, "745": 0,    # Ngọt
    "467": 1, "123": 1,    # Mặn
    "581": 2, "642": 2,    # Chua
    "934": 3, "314": 3,    # Đắng
}
LABEL2NAME = {0:"ngot",1:"man",2:"chua",3:"dang"}

def read_meta(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    meta = d["meta"]
    # meta có thể là JSON string, bytes, hoặc dict pickled
    if isinstance(meta, (np.ndarray,)) and meta.shape == ():
        meta = meta.item()
    if isinstance(meta, (bytes, np.bytes_)):
        meta = meta.decode("utf-8")
    if isinstance(meta, str):
        meta = json.loads(meta)
    assert isinstance(meta, dict), "meta phải là dict sau khi parse"
    return {
        "subject_id": meta.get("subject_id", npz_path.parent.name),
        "ma_mau": str(meta.get("ma_mau", "")),
        "fps": float(meta.get("fps", 0)),
        "T": int(meta.get("T", d["graph_seq"].shape[0] if "graph_seq" in d else 0)),
        "N_AU": int(meta.get("N_AU", d["graph_seq"].shape[1] if "graph_seq" in d else 0)),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Thư mục gốc chứa các thư mục P001/... gồm các .npz")
    ap.add_argument("--out-csv", default="split.csv", help="Đường dẫn CSV output")
    ap.add_argument("--val-size", type=float, default=0.2, help="Tỷ lệ validation theo subject")
    args = ap.parse_args()

    root = Path(args.root)
    rows = []
    for p in root.rglob("*.npz"):
        try:
            m = read_meta(p)
            code = m["ma_mau"].strip()
            if code not in CODE2LABEL:
                # Bỏ qua mẫu không thuộc 8 mã bạn đã chuẩn hoá
                continue
            label = CODE2LABEL[code]
            rows.append({
                "path": str(p.resolve()),
                "label": label,
                "label_name": LABEL2NAME[label],
                "subject_id": m["subject_id"],
                "code": code,
                "fps": m["fps"],
                "T": m["T"],
                "N_AU": m["N_AU"],
            })
        except Exception as e:
            print(f"[WARN] Bỏ qua {p}: {e}")

    df = pd.DataFrame(rows).sort_values(["subject_id","code","path"]).reset_index(drop=True)
    if df.empty:
        raise SystemExit("Không tìm thấy .npz hợp lệ!")

    # Tạo split theo subject để không rò rỉ
    groups = df["subject_id"].values
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=22)
    train_idx, val_idx = next(splitter.split(df, groups=groups))
    df["split"] = "train"
    df.loc[val_idx, "split"] = "val"

    # Thống kê nhanh
    print("Tổng:", len(df))
    print("Theo split:")
    print(df["split"].value_counts())
    print("Theo nhãn (toàn bộ):")
    print(df["label_name"].value_counts())

    df.to_csv(args.out_csv, index=False)
    print(f"Đã ghi {args.out_csv}")

if __name__ == "__main__":
    main()
