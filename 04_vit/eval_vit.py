import numpy as np, pandas as pd, xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score

df = pd.read_csv("../02_data/uk_dataset.csv")
meta = pd.read_csv("../02_data/vit_features_meta.csv")
FE = np.load("../02_data/vit_features.npy").astype("float32")
meta["fi"] = np.arange(len(meta))
df = df.merge(meta, on=["tile_id", "year", "month"], how="inner")
df = df[df.month.isin([2, 3, 4, 5, 6])].reset_index(drop=True)
print("rows with ViT features:", len(df))

BASE = ["month_sin", "month_cos", "elevation", "slope", "forest_frac"]
DYN = ["ndvi", "ndmi", "nbr", "temp_c", "precip_mm", "soil_moist"]
K = 16
VIT = [f"vit{i}" for i in range(K)]
SETS = {"Month + terrain": BASE, "+ ViT only": BASE + VIT,
        "+ indices/weather": BASE + DYN, "+ ViT + indices/weather": BASE + DYN + VIT}
res = {k: [] for k in SETS}
for ty in [2022, 2023, 2024, 2025]:
    tr, te = df[df.year < ty].copy(), df[df.year == ty].copy()
    pca = PCA(K, random_state=0).fit(FE[tr.fi])
    tr[VIT] = pca.transform(FE[tr.fi]); te[VIT] = pca.transform(FE[te.fi])
    pw = (tr.fire_next_month == 0).sum() / (tr.fire_next_month == 1).sum()
    for name, feats in SETS.items():
        m = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                              colsample_bytree=0.8, min_child_weight=5, scale_pos_weight=pw)
        m.fit(tr[feats], tr.fire_next_month)
        p = m.predict_proba(te[feats])[:, 1]
        res[name].append((average_precision_score(te.fire_next_month, p), roc_auc_score(te.fire_next_month, p)))
print()
for name, r in res.items():
    pr = [round(float(x[0]), 2) for x in r]
    print(f"{name:26s} PR-AUC mean {np.mean(pr):.3f} {pr}  ROC-AUC mean {np.mean([x[1] for x in r]):.3f}")
