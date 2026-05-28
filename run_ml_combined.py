#!/usr/bin/env python3
"""
Combined features: gERP-specific + existing v3 general features
===============================================================
Hypothesis: gERP features (frontal channels, late positivity) capture
*different* information from general features (connectivity, wavelets, etc.).
Combining both + MI selection might pick the best of each.

Setup:
  - gERP features  (489 cond-avg, quality-filtered)
  - v3 features    (962 general features from features_jar3_adv.csv)
  - Combined       (1451 total, MI selects best K)
  - Also try: gERP only top-5 features → simple LDA/Bayes

Models: XGB GPU, GradBoost (best from v2/v3)
K sweep: 5..35, iso=0.10, samp=[none, smote]
"""

import os, sys, warnings, datetime, glob
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import welch

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              f1_score, recall_score, confusion_matrix)
import xgboost as xgb
from imblearn.over_sampling import SMOTE

SEED = 42
np.random.seed(SEED)

EPOCH_DIR = 'output/epochs'
FEAT_CSV  = 'output/results/ml_jar3/features_jar3_adv.csv'
QUAL_CSV  = 'output/results/erp/erp_quality_flags.csv'
OUT_DIR   = 'output/results/ml_combined'
FIG_DIR   = 'output/figures/ml_combined'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'logs'), exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

SFREQ   = 100
TMIN    = -0.5
N_TIMES = 351
TIMES   = np.linspace(TMIN, TMIN + (N_TIMES-1)/SFREQ, N_TIMES)

CH_NAMES = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4',
            'O1','O2','F7','F8','T3','T4','Fz','Cz']
TASTE_CH = {'Fz':14,'Cz':15,'C3':4,'C4':5,'F3':2,'F4':3,
            'P3':6,'P4':7,'T3':12,'T4':13}


class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, msg):
        for s in self.streams: s.write(msg); s.flush()
    def flush(self):
        for s in self.streams: s.flush()


# ─── gERP feature extraction (condition-averaged) ────────────────────────────
def extract_gerp_cond_avg():
    t = TIMES
    rows = []
    for d in sorted(glob.glob(os.path.join(EPOCH_DIR, 'P*'))):
        subj = os.path.basename(d)
        npy = os.path.join(d, 'epochs_data.npy')
        ti  = os.path.join(d, 'trial_info.csv')
        if not os.path.exists(npy) or not os.path.exists(ti): continue
        epochs = np.load(npy).astype(np.float32)
        info   = pd.read_csv(ti)
        if len(epochs) != len(info): continue

        for cond, grp in info.groupby('condition'):
            avg = epochs[grp.index.tolist()].mean(axis=0)  # (16, 351)
            jar = grp['jar_group'].iloc[0]
            f = {'subject_id': subj, 'condition': int(cond), 'jar_group': jar}

            # Fine-grained 50ms bins 0→2s at taste channels
            for ch_name, ch_idx in TASTE_CH.items():
                sig = avg[ch_idx] * 1e6
                for i, t0 in enumerate(np.arange(0.0, 2.0, 0.05)):
                    t1 = t0 + 0.05
                    mask = (t >= t0) & (t < t1)
                    f[f'gerp_bin_{ch_name}_{int(t0*1000):04d}'] = float(sig[mask].mean()) if mask.any() else 0.0

            # Component features
            comps = {'P2':(0.20,0.35),'N400':(0.30,0.50),
                     'LP1':(0.50,0.70),'LP2':(0.70,1.00),'LP3':(1.00,1.50)}
            comp_ch = {'P2':['Cz','C3','C4','Fz'],'N400':['Cz','Fz','C3','C4','T3','T4'],
                       'LP1':['Cz','Fz','C3','C4','P3','P4'],
                       'LP2':['Cz','Fz','P3','P4'],'LP3':['Cz','Fz']}
            for cname,(t0,t1) in comps.items():
                mask = (t>=t0)&(t<=t1)
                for ch in comp_ch[cname]:
                    sig2 = avg[TASTE_CH[ch], mask]*1e6
                    if len(sig2) == 0: continue
                    f[f'gerp_{cname}_{ch}_mean'] = float(sig2.mean())
                    f[f'gerp_{cname}_{ch}_auc']  = float(np.trapz(sig2))

            # Asymmetry
            for r,l in [('F4','F3'),('C4','C3'),('P4','P3')]:
                for wn,(t0,t1) in [('N400',(0.3,0.5)),('LP',(0.5,1.0))]:
                    mask=(t>=t0)&(t<=t1)
                    rv=float(avg[TASTE_CH[r],mask].mean()*1e6)
                    lv=float(avg[TASTE_CH[l],mask].mean()*1e6)
                    f[f'gerp_asym_{r}m{l}_{wn}'] = rv - lv

            # LP/N400 ratio at Cz
            n400 = avg[15,(t>=0.3)&(t<=0.5)].mean()*1e6
            lp   = avg[15,(t>=0.5)&(t<=1.0)].mean()*1e6
            f['gerp_ratio_LP_N400_Cz'] = float(lp / (abs(n400)+1e-6))

            # Slope of LP at Fz (frontal — top MI feature)
            for ch_name,ch_idx in [('Fz',14),('F3',2)]:
                sig3 = avg[ch_idx]*1e6
                for t0,t1,label in [(0.30,0.60,'early'),(0.60,1.20,'late')]:
                    mask=(t>=t0)&(t<=t1)
                    if mask.sum()>=2:
                        seg=sig3[mask]
                        f[f'gerp_slope_{ch_name}_{label}'] = float(np.polyfit(np.arange(len(seg)),seg,1)[0])

            rows.append(f)

    df = pd.DataFrame(rows)
    gerp_cols = [c for c in df.columns if c.startswith('gerp_')]
    return df[['subject_id','condition','jar_group']+gerp_cols], gerp_cols


# ─── ML helpers ───────────────────────────────────────────────────────────────
def smote_safe(X, y):
    n_pos=(y==1).sum()
    if n_pos<2: return X,y
    k=min(5,n_pos-1)
    try: return SMOTE(random_state=SEED, k_neighbors=k).fit_resample(X,y)
    except: return X,y


def precompute_folds(X, y, g, iso=0.10):
    logo=LeaveOneGroupOut(); folds=[]
    for tr,te in logo.split(X,y,g):
        X_tr,y_tr=X[tr].copy(),y[tr].copy()
        X_te,y_te=X[te],y[te]
        if len(np.unique(y_tr))<2 or len(y_te)==0: continue
        if iso>0 and len(X_tr)>20:
            isof=IsolationForest(contamination=iso,random_state=SEED,n_jobs=-1)
            isof.fit(X_tr); km=isof.predict(X_tr)==1
            if (y_tr[km]==0).any() and (y_tr[km]==1).any():
                X_tr,y_tr=X_tr[km],y_tr[km]
        if len(np.unique(y_tr))<2: continue
        sc=StandardScaler()
        X_tr=np.nan_to_num(sc.fit_transform(X_tr))
        X_te=np.nan_to_num(sc.transform(X_te))
        mi=mutual_info_classif(X_tr,y_tr,random_state=SEED)
        order=np.argsort(mi)[::-1]
        folds.append({'X_tr':X_tr,'y_tr':y_tr,'X_te':X_te,'y_te':y_te,'mi_order':order})
    return folds


def eval_K(folds, K, model_factory, sampling='smote'):
    y_true_all,y_pred_all,proba_all=[],[],[]
    for f in folds:
        idx=f['mi_order'][:K]
        X_tr,y_tr=f['X_tr'][:,idx].copy(),f['y_tr'].copy()
        X_te=f['X_te'][:,idx]
        if sampling=='smote': X_tr,y_tr=smote_safe(X_tr,y_tr)
        m=model_factory(); m.fit(X_tr,y_tr)
        y_pred_all.extend(m.predict(X_te).tolist())
        y_true_all.extend(f['y_te'].tolist())
        if hasattr(m,'predict_proba'):
            proba_all.extend(m.predict_proba(X_te)[:,1].tolist())
        else:
            proba_all.extend([np.nan]*len(f['y_te']))
    if not y_true_all: return None
    yt,yp,pr=np.array(y_true_all),np.array(y_pred_all),np.array(proba_all)
    best_b,best_t=0.0,0.5
    if not np.isnan(pr).any():
        for thr in np.linspace(0.1,0.9,81):
            b=balanced_accuracy_score(yt,(pr>=thr).astype(int))
            if b>best_b: best_b=b; best_t=thr
    return {'accuracy':accuracy_score(yt,yp),
            'balanced_acc':balanced_accuracy_score(yt,yp),
            'f1_macro':f1_score(yt,yp,average='macro',zero_division=0),
            'rec_vua':recall_score(yt,yp,pos_label=1,zero_division=0),
            'rec_oth':recall_score(yt,yp,pos_label=0,zero_division=0),
            'oracle_bacc':round(float(best_b),4),'oracle_thr':round(float(best_t),2),
            'y_true':yt,'y_pred':yp}


def plot_confusion(yt,yp,title,path):
    cm=confusion_matrix(yt,yp)
    cmn=cm.astype(float)/cm.sum(axis=1,keepdims=True)
    ann=np.array([[f'{cm[i,j]}\n({cmn[i,j]*100:.0f}%)' for j in range(2)] for i in range(2)])
    fig,ax=plt.subplots(figsize=(5,4))
    sns.heatmap(cmn,annot=ann,fmt='',cmap='Blues',
                xticklabels=['Other','Vua_phai'],yticklabels=['Other','Vua_phai'],
                vmin=0,vmax=1,linewidths=0.5,ax=ax)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(title,fontsize=9,fontweight='bold')
    fig.tight_layout(); fig.savefig(path,dpi=150,bbox_inches='tight'); plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    ts=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_fh=open(os.path.join(OUT_DIR,'logs',f'run_{ts}.log'),'w')
    latest_fh=open(os.path.join(OUT_DIR,'run.log'),'w')
    sys.stdout=Tee(sys.__stdout__,log_fh,latest_fh)
    sys.stderr=Tee(sys.__stderr__,log_fh,latest_fh)

    print(f'[{datetime.datetime.now().isoformat(timespec="seconds")}] run_ml_combined')
    print('='*78)
    print('  Combined: gERP features + existing v3 features')
    print('='*78)
    t_start=datetime.datetime.now()

    # ── Load v3 general features ──────────────────────────────────────────
    print('\nLoading v3 general features...')
    df_v3=pd.read_csv(FEAT_CSV)
    qf=pd.read_csv(QUAL_CSV)[['subject_id','condition','quality_label','avg_snr','quality_score']]
    df_v3['condition']=df_v3['condition'].astype(int)
    qf['condition']=qf['condition'].astype(int)
    df_v3=df_v3.merge(qf,on=['subject_id','condition'],how='left')
    df_v3=df_v3[df_v3['quality_label']!='BAD'].copy()
    META_V3=['subject_id','condition','jar_group','quality_label','avg_snr','quality_score']
    v3_feat_cols=[c for c in df_v3.columns if c not in META_V3]
    print(f'  v3 features: {len(v3_feat_cols)}  n={len(df_v3)}')

    # ── Extract gERP features ─────────────────────────────────────────────
    print('Extracting gERP condition-averaged features...')
    df_gerp, gerp_cols = extract_gerp_cond_avg()
    df_gerp['condition']=df_gerp['condition'].astype(int)
    print(f'  gERP features: {len(gerp_cols)}  n={len(df_gerp)}')

    # ── Merge on (subject_id, condition) ─────────────────────────────────
    df_all=df_v3[['subject_id','condition','jar_group']+v3_feat_cols].merge(
        df_gerp[['subject_id','condition']+gerp_cols],
        on=['subject_id','condition'], how='inner')
    print(f'  Merged: n={len(df_all)}  total features={len(v3_feat_cols)+len(gerp_cols)}')

    all_feat_cols = v3_feat_cols + gerp_cols
    y = (df_all['jar_group'].values == 'Vua_phai').astype(int)
    g = df_all['subject_id'].values
    majority = max(int(y.sum()),int((y==0).sum()))/len(y)
    print(f'  pos={int(y.sum())}  neg={int((y==0).sum())}  majority={majority:.3f}')

    # Datasets to compare
    datasets = {
        'v3_only':      df_v3[v3_feat_cols].values.astype(float),
        'gerp_only':    df_all[gerp_cols].values.astype(float),
        'combined':     df_all[all_feat_cols].values.astype(float),
    }

    MODELS = {
        'XGB_gpu':   lambda: xgb.XGBClassifier(
            device='cuda', n_estimators=100, max_depth=4,
            learning_rate=0.05, subsample=0.8, scale_pos_weight=3,
            eval_metric='logloss', verbosity=0, random_state=SEED),
        'GradBoost': lambda: GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=3,
            subsample=0.8, random_state=SEED),
    }
    K_GRID   = [5,10,15,20,25,30,35]
    ISO      = 0.10

    rows=[]; best_row=None

    for ds_name, X_raw in datasets.items():
        X = np.nan_to_num(X_raw.astype(float), nan=0.0, posinf=0.0, neginf=0.0)
        keep = X.var(axis=0) > 1e-12
        X = X[:,keep]
        n_feat = X.shape[1]

        print(f'\n{"━"*60}')
        print(f'  Dataset: {ds_name}  ({n_feat} features)')

        folds=precompute_folds(X, y, g, ISO)
        print(f'  {len(folds)} folds prepared', flush=True)

        for sampling in ['none','smote']:
            for mname, mfac in MODELS.items():
                best_K_bacc=0.0; best_K_row=None
                for K in K_GRID:
                    res=eval_K(folds, K, mfac, sampling)
                    if res is None: continue
                    row={'dataset':ds_name,'model':mname,'sampling':sampling,'K':K,
                         'n_feat':n_feat,
                         'accuracy':    round(res['accuracy'],    4),
                         'balanced_acc':round(res['balanced_acc'],4),
                         'f1_macro':    round(res['f1_macro'],    4),
                         'rec_vua':     round(res['rec_vua'],     4),
                         'rec_oth':     round(res['rec_oth'],     4),
                         'oracle_bacc': res['oracle_bacc'],
                         'oracle_thr':  res['oracle_thr']}
                    rows.append(row)
                    if res['balanced_acc']>best_K_bacc:
                        best_K_bacc=res['balanced_acc']; best_K_row=row
                    if best_row is None or res['balanced_acc']>best_row['balanced_acc']:
                        best_row={**row,'y_true':res['y_true'],'y_pred':res['y_pred']}

                if best_K_row:
                    flag = ' ← NEW BEST!' if best_K_row['balanced_acc']==best_row['balanced_acc'] else ''
                    print(f'  {mname:<12} samp={sampling}  '
                          f'best K={best_K_row["K"]:<3}  '
                          f'bacc={best_K_row["balanced_acc"]:.4f}  '
                          f'oracle={best_K_row["oracle_bacc"]:.4f}  '
                          f'rec_vua={best_K_row["rec_vua"]:.4f}' + flag)

    # ── Save ──────────────────────────────────────────────────────────────
    df_res=pd.DataFrame(rows)
    df_res.to_csv(os.path.join(OUT_DIR,'results_combined.csv'),index=False)
    df_res.nlargest(20,'balanced_acc').to_csv(
        os.path.join(OUT_DIR,'top20_bacc.csv'),index=False)

    # Plot comparison
    fig,ax=plt.subplots(figsize=(12,5))
    ds_colors={'v3_only':'steelblue','gerp_only':'darkorange','combined':'green'}
    for ds_name,grp in df_res.groupby('dataset'):
        best=grp.groupby('K')['balanced_acc'].max()
        ax.plot(best.index, best.values, marker='o', lw=2,
                label=ds_name, color=ds_colors.get(ds_name,'gray'))
    ax.axhline(0.674,color='red',ls='--',lw=1.5,label='DL v2 best=0.674')
    ax.axhline(0.649,color='blue',ls='--',lw=1.5,label='ML v2 best=0.649')
    ax.axhline(majority,color='gray',ls=':',lw=1.0,label=f'majority={majority:.3f}')
    ax.set_xlabel('K'); ax.set_ylabel('Balanced Accuracy (max over models×samplings)')
    ax.set_title('Feature set comparison: v3 vs gERP vs Combined', fontweight='bold')
    ax.grid(alpha=0.3); ax.legend(fontsize=9); ax.set_ylim(0.40,0.85)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR,'dataset_comparison.png'),dpi=180,bbox_inches='tight')
    plt.close(fig)

    if best_row:
        plot_confusion(best_row['y_true'],best_row['y_pred'],
            f'{best_row["dataset"]} | {best_row["model"]} K={best_row["K"]} samp={best_row["sampling"]}\n'
            f'bacc={best_row["balanced_acc"]:.4f}  rec_vua={best_row["rec_vua"]:.4f}',
            os.path.join(FIG_DIR,'cm_best.png'))

    # Print dataset comparison table
    elapsed=(datetime.datetime.now()-t_start).seconds
    print(f'\n{"="*78}')
    print('  DATASET COMPARISON (best bacc per dataset)')
    print(f'{"="*78}')
    hdr=f'{"dataset":<14}{"model":<13}{"samp":<8}{"K":<5}{"acc":<8}{"bacc":<8}{"oracle_bacc":<13}{"rec_vua"}'
    print(hdr); print('-'*len(hdr))
    for ds in ['v3_only','gerp_only','combined']:
        sub=df_res[df_res['dataset']==ds]
        if sub.empty: continue
        best=sub.loc[sub['balanced_acc'].idxmax()]
        flag=' ← BEST' if best['balanced_acc']==best_row['balanced_acc'] else ''
        print(f'{ds:<14}{best["model"]:<13}{best["sampling"]:<8}'
              f'{int(best["K"]):<5}{best["accuracy"]:.4f}  {best["balanced_acc"]:.4f}  '
              f'{best["oracle_bacc"]:.4f}       {best["rec_vua"]:.4f}' + flag)

    if best_row:
        print(f'\n  OVERALL BEST: {best_row["dataset"]} {best_row["model"]} '
              f'K={best_row["K"]} samp={best_row["sampling"]}')
        print(f'    bacc={best_row["balanced_acc"]:.4f}  '
              f'oracle_bacc={best_row["oracle_bacc"]:.4f}  '
              f'rec_vua={best_row["rec_vua"]:.4f}')

    print(f'\n  Progression:')
    print(f'    ML v2 GradBoost:          bacc=0.649')
    print(f'    DL v2 ShallowConvNet:     bacc=0.674  ← previous best')
    if best_row:
        improved = best_row['balanced_acc'] > 0.674
        print(f'    Combined features:        bacc={best_row["balanced_acc"]:.3f}  '
              f'{"← IMPROVED!" if improved else "← similar"}')

    print(f'\n  Total: {elapsed}s ({elapsed//60}m {elapsed%60}s)')
    print(f'{"="*78}')
    print(f'[{datetime.datetime.now().isoformat(timespec="seconds")}] done.')
    log_fh.close(); latest_fh.close()


if __name__ == '__main__':
    main()
