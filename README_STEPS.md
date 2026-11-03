# Uttarakhand Forest Health & Fire Risk — Step by Step

## STEP 0: Setup (30 min)
1. Google Earth Engine account: https://earthengine.google.com -> Register (free, student/non-commercial)
2. console.cloud.google.com par ek project banao, project ID copy karo, usme Earth Engine API enable karo
3. Google Colab use karo (free GPU, koi install nahi) ya laptop par: pip install earthengine-api xgboost shap pandas scikit-learn
4. Terminal/Colab: earthengine authenticate

## STEP 1: Data export (01_gee_export/export_features.py)
- GEE_PROJECT mein apna ID daalo, run karo
- code.earthengine.google.com/tasks par 7 tasks (2019..2025) RUN dabao
- 10-40 min baad Drive/uk_forest mein CSV aayengi -> 02_data/ mein copy karo
- CHECK: print("Forest tiles") ~300-500 aana chahiye; CSV mein ndvi 0.2-0.9 ke beech ho

## STEP 2: Merge + label (03_model/merge_and_label.py)
- 7 CSV jodta hai, NDVI/NDMI/NBR ka change aur rolling mean banata hai
- LABEL: "agle mahine aag lagi?" (MODIS burned area se)
- Split: train 2019-23, val 2024, test 2025 (time wise, random nahi)

## STEP 3: XGBoost baseline + SHAP (03_model/train_xgboost.py)
- Pehle sirf tabular features se. ROC-AUC / PR-AUC dekho. shap_summary.png = XAI result
- Ye aapka pehla "real AI result" hai

## STEP 4 (next, main bhejunga): ViT
- GEE se har tile-month ki 224x224 Sentinel-2 image chips export
- Pretrained ViT (timm) se 768-dim features nikalna -> PCA -> XGBoost ke features mein jodna
- Compare: XGBoost-only vs ViT+XGBoost  (ye hi aapki novelty hai)
- Attention map visualize

## STEP 5: Dashboard
- Leaflet/Folium map: tile polygons + NDVI/NDMI/NBR + risk colour + SHAP top reasons

## STEP 6: India scale
- Uttarakhand code mein sirf boundary filter badlo (FAO GAUL level0 India / state-wise loop)
- Tile grid, features, model same. Cloud/compute zyada lagega -> state-by-state export karo
