# classic_baseline.py
import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score

def agg_feats(x):  # x: [T,N,F]
    # thống kê theo thời gian
    stats = []
    for fn in [np.mean, np.std, np.median]:
        stats.append(fn(x, axis=0).reshape(-1))   # [N*F]
    # IQR
    q75 = np.quantile(x, 0.75, axis=0)
    q25 = np.quantile(x, 0.25, axis=0)
    stats.append((q75 - q25).reshape(-1))
    # derivative stats
    dx = np.diff(x, axis=0)  # [T-1,N,F]
    stats.append(np.mean(dx, axis=0).reshape(-1))
    stats.append(np.std(dx, axis=0).reshape(-1))
    return np.concatenate(stats, axis=0)

df = pd.read_csv("split.csv")
X, y = [], []
for _, r in df.iterrows():
    d = np.load(Path(r["path"]), allow_pickle=True)
    x = d["graph_seq"]  # [T,N,F]
    X.append(agg_feats(x))
    y.append(int(r["label"]))
X = np.stack(X); y = np.array(y)

# Leave-One-Out trên từng video (hợp với cực ít mẫu)
loo = LeaveOneOut()
models = {
    "LogReg_L2": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0)),
    "SVM_RBF":   make_pipeline(StandardScaler(), SVC(kernel="rbf", C=2.0, gamma="scale"))
}
for name, clf in models.items():
    preds, trues = [], []
    for train_idx, test_idx in loo.split(X):
        clf.fit(X[train_idx], y[train_idx])
        p = clf.predict(X[test_idx])
        preds.extend(p.tolist()); trues.extend(y[test_idx].tolist())
    acc = accuracy_score(trues, preds)
    print(f"{name} LOO-ACC = {acc:.3f}")
