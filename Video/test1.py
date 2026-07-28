import numpy as np
d = np.load("/home/tran.xuan.tien@sun-asterisk.com/SunAI /EEG/Video/outputb2/P001/P001_467.npz", allow_pickle=True)
print(d["graph_seq"].shape)
print(d["graph_seq"].std(axis=0).mean(axis=0))  # trung bình std theo AU

import numpy as np, pandas as pd, glob, json
rows=[]
for f in glob.glob("/home/tran.xuan.tien@sun-asterisk.com/SunAI /EEG/Video/outputb2/**/*.npz", recursive=True):
    d=np.load(f,allow_pickle=True)
    x=d["graph_seq"]
    stds=x.std(axis=0).mean(axis=1) # std theo AU
    rows.append({"path":f,"mean_std":float(stds.mean()),"min_std":float(stds.min())})
pd.DataFrame(rows).sort_values("mean_std")

df = pd.DataFrame(rows)
if df.empty:
    print("❗Không thu được bản ghi nào. Kiểm tra lại ROOT hoặc cấu trúc thư mục.")
else:
    # Lưu ra CSV để soi tổng thể
    df_sorted = df.sort_values("mean_std").reset_index(drop=True)
    df_sorted.to_csv("npz_motion_stats.csv", index=False)
    print(df_sorted.head(10))  # Top 10 mẫu có mean_std thấp nhất (ít chuyển động)
    print("\n📄 Saved summary → npz_motion_stats.csv")

    # Một vài thống kê nhanh
    print("\n== Summary ==")
    print("mean_std (mean/min/max):",
          df['mean_std'].mean(), df['mean_std'].min(), df['mean_std'].max())
