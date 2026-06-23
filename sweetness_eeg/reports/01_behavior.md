# Stage 01 — Behavioral (Sensory) Analysis


Liking (9-point hedonic), sweetness Just-About-Right (JAR, 1–5, 3 = just right) and sweet aftertaste (1–5) for sucrose (S1) and sucralose (S2) at three iso-sweet intensities (~5 / 7.5 / 12 % sucrose), plus water.


- Participants: **23**, sample ratings: **322**


## Condition summary (mean ± SEM)


| condition     | substance   |   intensity |   n |   liking_mean |   liking_sem |   sweetness_jar_mean |   sweetness_jar_sem |   aftertaste_mean |   aftertaste_sem |
|:--------------|:------------|------------:|----:|--------------:|-------------:|---------------------:|--------------------:|------------------:|-----------------:|
| Sucralose-5   | Sucralose   |           5 |  46 |         5.022 |        0.259 |                1.891 |               0.109 |             1.826 |            0.129 |
| Sucralose-7.5 | Sucralose   |           7 |  46 |         5.804 |        0.203 |                2.87  |               0.141 |             2.848 |            0.139 |
| Sucralose-12  | Sucralose   |          12 |  46 |         5.065 |        0.335 |                4     |               0.158 |             3.913 |            0.164 |
| Sucrose-5     | Sucrose     |           5 |  46 |         5.87  |        0.198 |                2.783 |               0.124 |             2.413 |            0.141 |
| Sucrose-7.5   | Sucrose     |           7 |  46 |         5.826 |        0.243 |                3.217 |               0.135 |             3.022 |            0.151 |
| Sucrose-12    | Sucrose     |          12 |  46 |         5.065 |        0.375 |                4.065 |               0.15  |             3.761 |            0.165 |
| Water         | Water       |           0 |  46 |         4.217 |        0.342 |                1.109 |               0.064 |             1.087 |            0.061 |

## Liking, sweetness & aftertaste by condition


![Hedonic liking by condition](../figures/behavior/liking_box.png)

*Hedonic liking by condition*


![Sweetness JAR by condition](../figures/behavior/sweetness_jar_box.png)

*Sweetness JAR by condition*


![Sweet aftertaste by condition](../figures/behavior/aftertaste_box.png)

*Sweet aftertaste by condition*


![JAR category distribution](../figures/behavior/jar_distribution.png)

*JAR category distribution*


## Dose-response (sucrose vs sucralose)


![Liking vs intensity](../figures/behavior/dose_liking.png)

*Liking vs intensity*


![Sweetness JAR vs intensity](../figures/behavior/dose_sweetness.png)

*Sweetness JAR vs intensity*


![Aftertaste vs intensity](../figures/behavior/dose_aftertaste.png)

*Aftertaste vs intensity*


## Lingering aftertaste (key sucralose hypothesis)


The residual = *aftertaste − in-mouth sweetness*. A positive value means sweetness persists after expectoration — the lingering aftertaste sucralose is known for.


![Residual sweet aftertaste](../figures/behavior/aftertaste_residual.png)

*Residual sweet aftertaste*


## Sucrose vs sucralose contrasts (paired t-test, FDR-corrected)


**liking**


|   intensity |   n |     t |     p |   mean_sucrose |   mean_sucralose |   mean_diff |   p_fdr |
|------------:|----:|------:|------:|---------------:|-----------------:|------------:|--------:|
|           5 |  23 | 2.953 | 0.007 |          5.87  |            5.022 |       0.848 |   0.022 |
|           7 |  23 | 0.064 | 0.949 |          5.826 |            5.804 |       0.022 |   1     |
|          12 |  23 | 0     | 1     |          5.065 |            5.065 |       0     |   1     |

**sweetness_jar**


|   intensity |   n |     t |     p |   mean_sucrose |   mean_sucralose |   mean_diff |   p_fdr |
|------------:|----:|------:|------:|---------------:|-----------------:|------------:|--------:|
|           5 |  23 | 6.886 | 0     |          2.783 |            1.891 |       0.891 |   0     |
|           7 |  23 | 1.785 | 0.088 |          3.217 |            2.87  |       0.348 |   0.132 |
|          12 |  23 | 0.412 | 0.684 |          4.065 |            4     |       0.065 |   0.684 |

**aftertaste**


|   intensity |   n |      t |     p |   mean_sucrose |   mean_sucralose |   mean_diff |   p_fdr |
|------------:|----:|-------:|------:|---------------:|-----------------:|------------:|--------:|
|           5 |  23 |  3.839 | 0.001 |          2.413 |            1.826 |       0.587 |   0.003 |
|           7 |  23 |  0.89  | 0.383 |          3.022 |            2.848 |       0.174 |   0.383 |
|          12 |  23 | -1.232 | 0.231 |          3.761 |            3.913 |      -0.152 |   0.346 |

## Two-way repeated-measures ANOVA (substance × intensity)


**liking**


| Source                |     SS |   ddof1 |   ddof2 |    MS |     F |   p_unc |   p_GG_corr |   ng2 |   eps |
|:----------------------|-------:|--------:|--------:|------:|------:|--------:|------------:|------:|------:|
| substance             |  2.899 |       1 |      22 | 2.899 | 1.474 |   0.238 |       0.238 | 0.008 | 1     |
| intensity             | 12.938 |       2 |      44 | 6.469 | 1.625 |   0.208 |       0.216 | 0.034 | 0.64  |
| substance * intensity |  5.373 |       2 |      44 | 2.687 | 4.784 |   0.013 |       0.02  | 0.014 | 0.801 |

**sweetness_jar**


| Source                |     SS |   ddof1 |   ddof2 |     MS |      F |   p_unc |   p_GG_corr |   ng2 |   eps |
|:----------------------|-------:|--------:|--------:|-------:|-------:|--------:|------------:|------:|------:|
| substance             |  6.522 |       1 |      22 |  6.522 | 22.732 |   0     |       0     | 0.074 | 1     |
| intensity             | 66.743 |       2 |      44 | 33.371 | 87.191 |   0     |       0     | 0.45  | 0.913 |
| substance * intensity |  4.054 |       2 |      44 |  2.027 |  6.434 |   0.004 |       0.005 | 0.047 | 0.889 |

**aftertaste**


| Source                |     SS |   ddof1 |   ddof2 |     MS |      F |   p_unc |   p_GG_corr |   ng2 |   eps |
|:----------------------|-------:|--------:|--------:|-------:|-------:|--------:|------------:|------:|------:|
| substance             |  1.42  |       1 |      22 |  1.42  |  4.466 |   0.046 |       0.046 | 0.015 | 1     |
| intensity             | 67.895 |       2 |      44 | 33.947 | 58.911 |   0     |       0     | 0.42  | 0.842 |
| substance * intensity |  3.156 |       2 |      44 |  1.578 |  5.587 |   0.007 |       0.007 | 0.033 | 0.996 |
