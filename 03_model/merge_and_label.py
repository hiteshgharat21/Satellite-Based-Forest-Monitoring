"""STAGE 2: 7 CSV jodo, fire label banao (agle mahine aag lagegi ya nahi), temporal split."""
import glob, pandas as pd, numpy as np

df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("../02_data/uk_features_*.csv"))])
df = df.sort_values(["tile_id", "year", "month"]).reset_index(drop=True)
df["date"] = pd.to_datetime(dict(year=df.year, month=df.month, day=1))

# monsoon (Jun-Sep) mein S2 mein badal bahut -> missing NDVI ko tile ke andar interpolate karo
for c in ["ndvi", "ndmi", "nbr"]:
    df[c] = df.groupby("tile_id")[c].transform(lambda s: s.interpolate(limit=2))

# time features: pichhle mahine se change aur 3 mahine ka rolling mean
g = df.groupby("tile_id")
for c in ["ndvi", "ndmi", "nbr"]:
    df[f"{c}_delta"] = g[c].diff()
    df[f"{c}_roll3"] = g[c].transform(lambda s: s.rolling(3, min_periods=1).mean())
df["precip_roll3"] = g["precip_mm"].transform(lambda s: s.rolling(3, min_periods=1).sum())
df["month_sin"] = np.sin(2 * np.pi * df.month / 12); df["month_cos"] = np.cos(2 * np.pi * df.month / 12)

# LABEL: agle mahine tile mein aag (burned_frac > 0) hui?
df["fire_next_month"] = (g["burned_frac"].shift(-1) > 0).astype("float")
df = df.dropna(subset=["fire_next_month", "ndvi"])

# SPLIT TIME ke hisab se (random nahi!) -> data leakage se bachne ke liye
train = df[df.year <= 2023]; val = df[df.year == 2024]; test = df[df.year == 2025]
print("rows:", len(train), len(val), len(test), "| fire rate:", df.fire_next_month.mean().round(4))
df.to_csv("../02_data/uk_dataset.csv", index=False)
