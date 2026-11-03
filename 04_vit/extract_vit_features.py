import glob, re, numpy as np, pandas as pd, torch, timm, rasterio
import torch.nn.functional as F
from tqdm import tqdm

DATA    = "../02_data/uk_dataset.csv"
RGB_DIR = "/content/drive/MyDrive/uk_rgb"
HALF, N = 0.05, 100

tiles = pd.read_csv(DATA)[["tile_id", "cx", "cy"]].drop_duplicates("tile_id").reset_index(drop=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0).to(dev).eval()
cfg = timm.data.resolve_data_config({}, model=model)
mean = torch.tensor(cfg["mean"], device=dev).view(1, 3, 1, 1)
std = torch.tensor(cfg["std"], device=dev).view(1, 3, 1, 1)

feats, meta = [], []
files = sorted(glob.glob(f"{RGB_DIR}/uk_rgb_*.tif"))
print("files:", len(files), "| device:", dev)
for f in tqdm(files):
    year, month = map(int, re.search(r"uk_rgb_(\d{4})_(\d{2})", f).groups())
    chips, nod = [], []
    with rasterio.open(f) as src:
        arr = src.read()
        for _, t in tiles.iterrows():
            r, c = src.index(t.cx - HALF, t.cy + HALF)
            ch = arr[:, max(r, 0):r + N, max(c, 0):c + N]
            if ch.shape != (3, N, N):
                ch = np.zeros((3, N, N), np.uint8)
            chips.append(ch); nod.append(float((ch.sum(0) == 0).mean()))
    chips = np.stack(chips)
    out = []
    with torch.no_grad():
        for i in range(0, len(chips), 64):
            x = torch.from_numpy(chips[i:i + 64]).to(dev).float() / 255.0
            x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
            out.append(model((x - mean) / std).float().cpu().numpy())
    feats.append(np.concatenate(out).astype(np.float16))
    meta.append(tiles[["tile_id"]].assign(year=year, month=month, nodata_frac=nod))

np.save("../02_data/vit_features.npy", np.concatenate(feats))
pd.concat(meta, ignore_index=True).to_csv("../02_data/vit_features_meta.csv", index=False)
print("saved", np.concatenate(feats).shape)
