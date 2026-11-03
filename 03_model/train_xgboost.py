"""STAGE 3: XGBoost baseline + SHAP. """
import pandas as pd, xgboost as xgb, shap
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

df = pd.read_csv("../02_data/uk_dataset.csv")
FEATS = ["ndvi","ndmi","nbr","ndvi_delta","ndmi_delta","nbr_delta","ndvi_roll3","ndmi_roll3","nbr_roll3",
         "temp_c","precip_mm","precip_roll3","soil_moist","elevation","slope","forest_frac","month_sin","month_cos"]
tr, va, te = df[df.year<=2023], df[df.year==2024], df[df.year==2025]

pos_w = (tr.fire_next_month==0).sum() / max((tr.fire_next_month==1).sum(), 1)   # aag kam hoti hai -> imbalance
model = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, scale_pos_weight=pos_w, eval_metric="aucpr",
                          early_stopping_rounds=30)
model.fit(tr[FEATS], tr.fire_next_month, eval_set=[(va[FEATS], va.fire_next_month)], verbose=False)

p = model.predict_proba(te[FEATS])[:, 1]
print("TEST ROC-AUC:", roc_auc_score(te.fire_next_month, p).round(3))
print("TEST PR-AUC :", average_precision_score(te.fire_next_month, p).round(3))
print(classification_report(te.fire_next_month, p > 0.5))

exp = shap.TreeExplainer(model); sv = exp.shap_values(te[FEATS])
shap.summary_plot(sv, te[FEATS], show=False); import matplotlib.pyplot as plt
plt.savefig("shap_summary.png", bbox_inches="tight", dpi=150)
te.assign(risk=p).to_csv("../02_data/uk_predictions.csv", index=False)
