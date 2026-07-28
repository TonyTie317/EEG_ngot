# dataset_stgcn.py
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
# N =15AU,  F=10 đặc trưng, T số frames
def _temporal_fix(x_np, t_fix: int, mode: str = "center"):
    """
    x_np: [T, N, F]
    Trả về [T_fix, N, F]
    - mode='center': nếu dài hơn thì cắt giữa; nếu ngắn hơn thì pad lặp frame cuối
    """
    T, N, F = x_np.shape
    if t_fix is None or t_fix <= 0:
        return x_np  # không ép độ dài
    if T == t_fix:
        return x_np
    if T > t_fix:
        # center-crop
        start = (T - t_fix) // 2
        end = start + t_fix
        return x_np[start:end]
    else:
        # pad bằng frame cuối
        pad_len = t_fix - T
        last = x_np[-1:,...]                  # [1, N, F]
        pad = np.repeat(last, pad_len, axis=0)  # [pad_len, N, F] 
        return np.concatenate([x_np, pad], axis=0)

class AUSequenceDataset(Dataset):
    """
    Đọc split.csv, mỗi hàng là 1 .npz
    Trả về:
      x: [N, T_fix, F]
      adj: [N, N]
      y: long
    """
    def __init__(self, csv_rows, t_fix: int = None, fix_mode: str = "center"):
        self.rows = csv_rows
        self.t_fix = t_fix
        self.fix_mode = fix_mode

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        npz_path = Path(row["path"])
        d = np.load(npz_path, allow_pickle=True)
        x = d["graph_seq"]          # [T, N, F]
        adj = d["adj"]              # [N, N]

        # ép độ dài thời gian
        x = _temporal_fix(x, self.t_fix, self.fix_mode)   # [T_fix, N, F] nếu t_fix!=None

        # sang tensor [N, T, F]
        x = torch.from_numpy(x).float().permute(1, 0, 2).contiguous()
        adj = torch.from_numpy(adj).float()
        y = torch.tensor(int(row["label"]), dtype=torch.long)
        return x, adj, y, row["subject_id"], row["code"]
