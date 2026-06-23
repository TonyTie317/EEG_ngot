# Stage 11 — Per-channel, within-substance dose-response


Unlike Stage 05 (sucrose **vs** sucralose, pooled into Frontal/Gustatory ROIs), this stage analyses **each substance on its own** and resolves the **individual channels** (all 16 electrodes). For every substance × band × channel we test how relative band power changes across the three perceived intensities (~5 / 7.5 / 12 %):

- **rmANOVA** over intensity — omnibus "is there any dose effect?" (F, p).
- **Per-subject linear slope** of rel-power vs intensity, one-sample t-test vs 0 — the *direction* (positive ⇒ power rises with concentration).

p-values are **FDR-corrected across the 16 channels within each band**. Channels surviving FDR are circled on the topographies / starred in the heatmaps.


## Dose slope topographies (direction of the effect)


![Per-subject slope t-value of relative band power vs intensity — rows = bands, cols = substance. Red = power increases with concentration, blue = decreases. Circled = FDR p<0.05.](../figures/channel_dose/topomap_dose_slope_grid.png)

*Per-subject slope t-value of relative band power vs intensity — rows = bands, cols = substance. Red = power increases with concentration, blue = decreases. Circled = FDR p<0.05.*


## Omnibus dose effect (rmANOVA F)


![rmANOVA F-statistic for the intensity main effect per channel (circled = FDR p<0.05).](../figures/channel_dose/topomap_dose_anovaF_grid.png)

*rmANOVA F-statistic for the intensity main effect per channel (circled = FDR p<0.05).*


## Channel × band slope heatmaps


![Sucrose: dose slope t per channel × band](../figures/channel_dose/heatmap_dose_slope_sucrose.png)

*Sucrose: dose slope t per channel × band*


![Sucralose: dose slope t per channel × band](../figures/channel_dose/heatmap_dose_slope_sucralose.png)

*Sucralose: dose slope t per channel × band*


## Strongest channel-level dose effects


Top channel × band combinations by raw slope p-value (both substances shown for context).


| substance   | band   | channel   |   n | F   | p_anova   |   p_anova_fdr |   slope |   t_slope |   p_slope |   p_slope_fdr |
|:------------|:-------|:----------|----:|:----|:----------|--------------:|--------:|----------:|----------:|--------------:|
| Sucralose   | theta  | F7        |  23 |     |           |             1 |  -0.001 |    -2.603 |     0.016 |         0.26  |
| Sucralose   | alpha  | T7        |  23 |     |           |             1 |   0.001 |     2.259 |     0.034 |         0.546 |
| Sucralose   | theta  | P7        |  23 |     |           |             1 |  -0.001 |    -2.222 |     0.037 |         0.268 |
| Sucralose   | theta  | F4        |  23 |     |           |             1 |  -0.002 |    -2.051 |     0.052 |         0.268 |
| Sucralose   | theta  | C3        |  23 |     |           |             1 |  -0.001 |    -1.927 |     0.067 |         0.268 |
| Sucralose   | beta   | Fp1       |  23 |     |           |             1 |   0.002 |     1.698 |     0.104 |         0.483 |
| Sucralose   | beta   | Fp2       |  23 |     |           |             1 |   0.002 |     1.65  |     0.113 |         0.483 |
| Sucralose   | theta  | Fp1       |  23 |     |           |             1 |  -0.001 |    -1.647 |     0.114 |         0.364 |

![F7 theta (Sucralose)](../figures/channel_dose/dose_F7_theta.png)

*F7 theta (Sucralose)*


![T7 alpha (Sucralose)](../figures/channel_dose/dose_T7_alpha.png)

*T7 alpha (Sucralose)*


![P7 theta (Sucralose)](../figures/channel_dose/dose_P7_theta.png)

*P7 theta (Sucralose)*


![F4 theta (Sucralose)](../figures/channel_dose/dose_F4_theta.png)

*F4 theta (Sucralose)*


![C3 theta (Sucralose)](../figures/channel_dose/dose_C3_theta.png)

*C3 theta (Sucralose)*


![Fp1 beta (Sucralose)](../figures/channel_dose/dose_Fp1_beta.png)

*Fp1 beta (Sucralose)*


![Fp2 beta (Sucralose)](../figures/channel_dose/dose_Fp2_beta.png)

*Fp2 beta (Sucralose)*


![Fp1 theta (Sucralose)](../figures/channel_dose/dose_Fp1_theta.png)

*Fp1 theta (Sucralose)*


## FDR-significant dose channels (summary)


*No channel survived FDR correction for a linear dose effect in either substance — dose modulation of band power is weak at the single-channel level (consistent with the weak ROI-level contrasts in Stage 05).*

