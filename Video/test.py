import numpy as np, json

data = np.load("/home/tran.xuan.tien@sun-asterisk.com/SunAI /EEG/Video/outputb2/P001/P001_213.npz", allow_pickle=True)
print(data.files)

print(data['graph_seq'].shape)
print(data['adj'].shape)
print(json.loads(data['meta'].item()))
X = data["graph_seq"]         # [T, N, F]
A = data["adj"]               # [N, N]
meta = json.loads(data["meta"].item())
T, N, F = X.shape

print("Duration (s) =", meta["T"]/meta["fps"])  # ~11.0

# 1) Kiểm tra adj đối xứng & có self-loop
print("Adj symmetric? ", np.allclose(A, A.T))
print("Diag (self-loops) =", np.diag(A))

# 2) Phạm vi các feature chính
cx, cy, cz, vis, area, aspect, dcx, dcy, darea, daspect = range(10)
print("cx [min,max] =", float(X[..., cx].min()), float(X[..., cx].max()))
print("cy [min,max] =", float(X[..., cy].min()), float(X[..., cy].max()))
print("vis [min,max] =", float(X[..., vis].min()), float(X[..., vis].max()))
print("area>0? ", bool((X[..., area] > 0).all()))

# 3) Đạo hàm nên có trung bình gần 0
for i, name in enumerate(meta["feature_names"][6:], start=6):
    print(name, "mean≈0?", abs(X[..., i].mean()) < 1e-3, "mean=", float(X[..., i].mean()))


import matplotlib.pyplot as plt

plt.imshow(A, interpolation="nearest")
plt.title("Adjacency (15x15)")
plt.colorbar(); plt.xlabel("AU"); plt.ylabel("AU")
plt.show()

au_names = meta["au_nodes"]
k = au_names.index("upper_lip")
t = np.arange(T) / meta["fps"]

plt.plot(t, X[:, k, cy], label="upper_lip.cy")
plt.plot(t, X[:, k, dcy], label="upper_lip.dcy", alpha=0.7)
plt.legend(); plt.xlabel("seconds"); plt.ylabel("value"); plt.title("Upper lip dynamics")
plt.show()