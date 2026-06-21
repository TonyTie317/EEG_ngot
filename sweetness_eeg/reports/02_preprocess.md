# Stage 02 — Preprocessing & Epoching


Pipeline: pick 16 EEG channels (T3/T4/T5/T6 → T7/T8/P7/P8) → average reference → notch 49 Hz → band-pass 0.5–45 Hz. Each trial = 10 s tasting window (1000 samples @ 100 Hz). Peak-to-peak rejection: 1000 µV.


- Subjects with usable epochs: **23**
- Total epochs: **320** (max possible = 322)


## Epochs kept per subject × condition


| subject   |   Water |   Sucrose-5 |   Sucrose-7.5 |   Sucrose-12 |   Sucralose-5 |   Sucralose-7.5 |   Sucralose-12 |   total |
|:----------|--------:|------------:|--------------:|-------------:|--------------:|----------------:|---------------:|--------:|
| P001      |       2 |           2 |             2 |            2 |             2 |               2 |              2 |      14 |
| P002      |       2 |           2 |             2 |            2 |             2 |               2 |              2 |      14 |
| P003      |       2 |           2 |             2 |            2 |             2 |               2 |              2 |      14 |
| P004      |       2 |           2 |             2 |            2 |             2 |               2 |              2 |      14 |
| P005      |       2 |           2 |             2 |            2 |             2 |               2 |              2 |      14 |
| P006      |       2 |           2 |             2 |            2 |             2 |               2 |              2 |      14 |
| P007      |       2 |           2 |             2 |            2 |             2 |               2 |              2 |      14 |
| P008      |       2 |           2 |             2 |            2 |             2 |               2 |              2 |      14 |
| P009      |       2 |           2 |             2 |            2 |             2 |               2 |              2 |      14 |
| P010      |       2 |           2 |             2 |            2 |             2 |               2 |              2 |      14 |
| P011      |       2 |           2 |             2 |            2 |             2 |               2 |              2 |      14 |
| P012      |       2 |           2 |             2 |            2 |             2 |               1 |              2 |      13 |
| P014      |       2 |           2 |             2 |            2 |             2 |               2 |              2 |      14 |
| P015      |       2 |           2 |             2 |            2 |             2 |               2 |              2 |      14 |
| P016      |       2 |           2 |             2 |            2 |             2 |               2 |              2 |      14 |
| P017      |       2 |           2 |             2 |            2 |             2 |               2 |              2 |      14 |
| P019      |       2 |           2 |             2 |            2 |             2 |               2 |              2 |      14 |
| P020      |       2 |           2 |             2 |            2 |             2 |               2 |              2 |      14 |
| P021      |       2 |           2 |             2 |            2 |             1 |               2 |              2 |      13 |
| P022      |       2 |           2 |             2 |            2 |             2 |               2 |              2 |      14 |
| P023      |       2 |           2 |             2 |            2 |             2 |               2 |              2 |      14 |
| P024      |       2 |           2 |             2 |            2 |             2 |               2 |              2 |      14 |
| P025      |       2 |           2 |             2 |            2 |             2 |               2 |              2 |      14 |

## Epoch counts per condition (pooled)


| condition     |   n_epochs |
|:--------------|-----------:|
| Water         |         46 |
| Sucrose-5     |         46 |
| Sucrose-7.5   |         46 |
| Sucrose-12    |         46 |
| Sucralose-5   |         45 |
| Sucralose-7.5 |         45 |
| Sucralose-12  |         46 |
