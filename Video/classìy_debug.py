#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, numpy as np, pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

LABEL2NAME = {0:"ngot",1:"man",2:"chua",3:"dang"}

def agg_feats(x):  # x: [T,N,F]
    # Thống kê theo thời gian + đạo hàm
    stats = []
    for fn in [np.mean, np.std, np.median]:
        stats.append(fn(x, axis=0).reshape(-1))   # [N*F]
    q75 = np.quantile(x, 0.75, axis=0)
    q25 = np.quantile(x, 0.25, axis=0)
    stats.append((q75 - q25).reshape(-1))
    dx = np.diff(x, axis=0)  # [T-1,N,F]
    stats.append(np.mean(dx, axis=0).reshape(-1))
    stats.append(np.std(dx, axis=0).reshape(-1))
    v = np.concatenate(stats, axis=0)
    return np.nan_to_num(v, copy=False)

def load_data():
    df = pd.read_csv("split.csv")
    # Chỉ giữ 4 nhãn 0..3
    df = df[df["label"].isin([0,1,2,3])].reset_index(drop=True)

    X, y, names = [], [], []
    for _, r in df.iterrows():
        d = np.load(Path(r["path"]), allow_pickle=True)
        x = d["graph_seq"]  # [T,N,F]
        X.append(agg_feats(x))
        y.append(int(r["label"]))
        names.append(f'{Path(r["path"]).name}::{r["label_name"]}')
    X = np.stack(X); y = np.array(y)
    return df, X, y, names

def print_basic_stats(df, X, y):
    print("== Label distribution ==")
    print(df["label_name"].value_counts(), "\n")
    print("== Split counts ==")
    if "split" in df.columns:
        print(df.groupby(["split","label_name"]).size(), "\n")

    # Kiểm tra phương sai đặc trưng
    stds = X.std(axis=0)
    zero_var = int((stds < 1e-8).sum())
    print(f"Features: {X.shape[1]} | zero-variance: {zero_var}")
    if zero_var > 0:
        print("⚠️ Có đặc trưng phương sai ~0 → có thể dữ liệu hằng số / preprocessing lỗi")

def overfit_test(X, y):
    print("\n== Overfit test (train & evaluate trên toàn bộ) ==")
    models = {
        "LogReg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=10.0, multi_class='auto')),
        "SVM_RBF": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0, gamma="scale")),
        "LDA": make_pipeline(StandardScaler(with_mean=True, with_std=True), LDA()),
        "KNN5": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    }
    for name, clf in models.items():
        clf.fit(X, y)
        pred = clf.predict(X)
        acc = accuracy_score(y, pred)
        print(f"{name} train-ACC = {acc:.3f}")

def shuffle_label_test(X, y):
    print("\n== Shuffle-label sanity check ==")
    rng = np.random.default_rng(42)
    y_shuffle = y.copy()
    rng.shuffle(y_shuffle)
    clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10.0, gamma="scale"))
    loo = LeaveOneOut()
    preds, trues = [], []
    for tr, te in loo.split(X):
        clf.fit(X[tr], y_shuffle[tr])
        preds.extend(clf.predict(X[te]).tolist())
        trues.extend(y_shuffle[te].tolist())
    acc = accuracy_score(trues, preds)
    print(f"SVM_RBF LOO-ACC (labels shuffled) = {acc:.3f}  (kỳ vọng ~0.25 nếu 4 lớp)")

def loo_eval(name, clf, X, y, names):
    loo = LeaveOneOut()
    preds, trues, test_names = [], [], []
    for tr, te in loo.split(X):
        clf.fit(X[tr], y[tr])
        p = clf.predict(X[te])
        preds.extend(p.tolist())
        trues.extend(y[te].tolist())
        test_names.append(names[te[0]])
    acc = accuracy_score(trues, preds)
    print(f"{name} LOO-ACC = {acc:.3f}")
    # Confusion matrix + report
    cm = confusion_matrix(trues, preds, labels=[0,1,2,3])
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    print(classification_report(trues, preds, target_names=[LABEL2NAME[i] for i in [0,1,2,3]], digits=3))
    # Liệt kê dự đoán từng mẫu
    print("\n== Per-sample LOO predictions ==")
    for nm, t, p in zip(test_names, trues, preds):
        print(f"{nm} | true={LABEL2NAME[t]} pred={LABEL2NAME[p]}")

def main():
    df, X, y, names = load_data()
    print_basic_stats(df, X, y)

    # Overfit test: nếu ở đây vẫn thấp -> có vấn đề nặng (label/mapping/feature hằng số)
    overfit_test(X, y)

    # Shuffle-label sanity: xác nhận quy trình đánh giá hợp lý
    shuffle_label_test(X, y)

    print("\n== LOO baselines ==")
    # PCA + SVM: giảm chiều trước khi SVM (hợp với ít mẫu)
    n_comp = max(1, min(X.shape[0]-1, 32))
    loo_eval("PCA+SVM_RBF",
             make_pipeline(StandardScaler(), PCA(n_components=n_comp), SVC(kernel="rbf", C=5.0, gamma="scale")),
             X, y, names)

    # LDA: tốt cho ít mẫu, giả định gaussian theo lớp
    loo_eval("LDA",
             make_pipeline(StandardScaler(with_mean=True, with_std=True), LDA()),
             X, y, names)

    # KNN: phi tham số, tham khảo
    loo_eval("KNN5",
             make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
             X, y, names)

if __name__ == "__main__":
    main()
