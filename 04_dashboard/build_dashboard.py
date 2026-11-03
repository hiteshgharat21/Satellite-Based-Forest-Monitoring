import json, numpy as np, pandas as pd, xgboost as xgb, shap
from sklearn.metrics import roc_auc_score, average_precision_score

df = pd.read_csv("../02_data/uk_dataset.csv")
BASE = ["month_sin", "month_cos", "elevation", "slope", "forest_frac"]
DYN = ["ndvi", "ndmi", "nbr", "temp_c", "precip_mm", "soil_moist"]
FEATS = BASE + DYN
NAMES = {"month_sin": "Season (month)", "month_cos": "Season (month)", "elevation": "Elevation", "slope": "Slope",
         "forest_frac": "Forest fraction", "ndvi": "NDVI (greenness)", "ndmi": "NDMI (moisture)", "nbr": "NBR (burn index)",
         "temp_c": "Temperature", "precip_mm": "Rainfall", "soil_moist": "Soil moisture"}

tr, te = df[df.year <= 2024], df[df.year == 2025].copy()
model = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=5)
model.fit(tr[FEATS], tr.fire_next_month)
te["risk"] = model.predict_proba(te[FEATS])[:, 1]
print("2025  ROC-AUC %.3f  PR-AUC %.3f" % (roc_auc_score(te.fire_next_month, te.risk),
                                         average_precision_score(te.fire_next_month, te.risk)))

sv = shap.TreeExplainer(model).shap_values(te[FEATS])
te = te.reset_index(drop=True)

months = {}
tot_hit = tot_fire = 0
for m, g in te.groupby("month"):
    k = max(1, int(round(0.10 * len(g))))
    top = g.sort_values("risk", ascending=False).head(k)
    hit, fire = int(top.fire_next_month.sum()), int(g.fire_next_month.sum())
    tot_hit += hit; tot_fire += fire
    months[int(m)] = {"top_k": k, "hit": hit, "fire": fire}
print("Top-10%% tiles/month ne 2025 ki %d me se %d aag wali tile-months pakdi" % (tot_fire, tot_hit))

tiles = {}
for i, r in te.iterrows():
    order = np.argsort(-np.abs(sv[i]))
    reasons, seen = [], set()
    for j in order:
        nm = NAMES[FEATS[j]]
        if nm in seen:
            continue
        seen.add(nm)
        reasons.append([nm, round(float(sv[i][j]), 3)])
        if len(reasons) == 3:
            break
    t = tiles.setdefault(r.tile_id, {"id": r.tile_id, "cx": round(float(r.cx), 3), "cy": round(float(r.cy), 3),
                                     "elev": round(float(r.elevation)), "m": {}})
    t["m"][int(r.month)] = [round(float(r.risk), 4), int(r.fire_next_month),
                            None if pd.isna(r.ndvi) else round(float(r.ndvi), 3),
                            None if pd.isna(r.ndmi) else round(float(r.ndmi), 3),
                            None if pd.isna(r.nbr) else round(float(r.nbr), 3), reasons]
payload = {"tiles": list(tiles.values()), "months": months}

HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>Uttarakhand Forest Fire Risk</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
body{margin:0;font-family:system-ui,Segoe UI,Arial,sans-serif}#map{position:absolute;inset:0}
.panel{position:absolute;z-index:1000;top:10px;right:10px;width:290px;background:#fff;padding:12px 14px;border-radius:10px;box-shadow:0 2px 12px #0004;font-size:13px}
.panel h3{margin:0 0 8px;font-size:15px}.panel label{display:block;margin:8px 0 2px;font-weight:600}
.panel select{width:100%;padding:5px}.stat{margin-top:10px;padding:8px;background:#f3f6f4;border-radius:6px}
.legend{margin-top:10px}.bar{height:10px;border-radius:5px;background:linear-gradient(90deg,#2e7d32,#fdd835,#d32f2f)}
.bar.v{background:linear-gradient(90deg,#8d6e63,#fdd835,#2e7d32)}.lg{display:flex;justify-content:space-between;font-size:11px;color:#555}
.pop b{display:block;margin-top:6px}</style></head><body><div id="map"></div>
<div class="panel"><h3>Uttarakhand Forest Fire Risk (2025)</h3>
<label>Month (prediction for next month)</label><select id="mo"></select>
<label>Layer</label><select id="ly"><option value="0">Predicted fire risk</option><option value="1">Actual fire next month</option>
<option value="2">NDVI</option><option value="3">NDMI</option><option value="4">NBR</option></select>
<div class="stat" id="stat"></div><div class="legend"><div class="bar" id="bar"></div><div class="lg"><span id="lo"></span><span id="hi"></span></div></div>
<div style="margin-top:8px;color:#777;font-size:11px">Model: XGBoost trained on 2019-2024, tested on unseen 2025. Click a tile for SHAP reasons.</div></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const D=__DATA__;const MN=["","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
const map=L.map('map').setView([30.1,79.2],8);
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',{attribution:'Tiles &copy; Esri'}).addTo(map);
const mo=document.getElementById('mo'),ly=document.getElementById('ly');
Object.keys(D.months).forEach(m=>{const o=document.createElement('option');o.value=m;o.textContent=MN[m]+" 2025";mo.appendChild(o)});
function lerp(a,b,t){return a.map((x,i)=>Math.round(x+(b[i]-x)*t))}
function ramp(t,rev){t=Math.max(0,Math.min(1,t));if(rev)t=1-t;const g=[46,125,50],y=[253,216,53],r=[211,47,47];
 const c=t<.5?lerp(g,y,t*2):lerp(y,r,(t-.5)*2);return`rgb(${c})`}
function vramp(t){t=Math.max(0,Math.min(1,t));const b=[141,110,99],y=[253,216,53],g=[46,125,50];
 const c=t<.5?lerp(b,y,t*2):lerp(y,g,(t-.5)*2);return`rgb(${c})`}
const RANGE={2:[0,0.9],3:[-0.1,0.5],4:[0,0.6]};let rects=[];
function popup(t,m){const d=t.m[m];const rs=d[5].map(r=>`<div>${r[1]>0?'&#9650;':'&#9660;'} ${r[0]} <span style="color:#777">(${r[1]>0?'+':''}${r[1]})</span></div>`).join('');
 return`<div class="pop"><b>${t.id}</b>Elevation: ${t.elev} m<br>Risk: <b style="display:inline">${(d[0]*100).toFixed(1)}%</b><br>
 Actual fire next month: ${d[1]?'<b style="display:inline;color:#d32f2f">YES</b>':'no'}<br>NDVI ${d[2]} | NDMI ${d[3]} | NBR ${d[4]}
 <b>Why (SHAP, top 3)</b>${rs}<span style="color:#777;font-size:11px">&#9650; raises risk, &#9660; lowers risk</span></div>`}
function draw(){rects.forEach(r=>map.removeLayer(r));rects=[];const m=mo.value,l=+ly.value;const H=0.05;
 D.tiles.forEach(t=>{const d=t.m[m];if(!d)return;let c;
  if(l==0)c=ramp(d[0]/0.4);else if(l==1)c=d[1]?'#d32f2f':'#c8e6c9';
  else{const v=d[l];if(v===null){c='#bbb'}else{const [a,b]=RANGE[l];c=vramp((v-a)/(b-a))}}
  const r=L.rectangle([[t.cy-H,t.cx-H],[t.cy+H,t.cx+H]],{color:'#333',weight:.4,fillColor:c,fillOpacity:.75}).addTo(map);
  r.bindPopup(popup(t,m));rects.push(r)});
 const s=D.months[m];document.getElementById('stat').innerHTML=`<b>${MN[m]} 2025</b><br>Actual fire tiles next month: <b>${s.fire}</b><br>
 Top 10% risky tiles (${s.top_k}) caught: <b>${s.hit}</b> of ${s.fire}`;
 const bar=document.getElementById('bar');bar.className='bar'+(l>=2?' v':'');
 document.getElementById('lo').textContent=l==0?'low risk':l==1?'':RANGE[l][0];document.getElementById('hi').textContent=l==0?'40%+ risk':l==1?'':RANGE[l][1]}
mo.onchange=ly.onchange=draw;mo.value=Object.keys(D.months).includes("4")?"4":Object.keys(D.months)[0];draw();
const b=L.latLngBounds(D.tiles.map(t=>[t.cy,t.cx]));map.fitBounds(b.pad(0.1));
</script></body></html>"""
open("uk_dashboard.html", "w", encoding="utf-8").write(HTML.replace("__DATA__", json.dumps(payload)))
print("saved uk_dashboard.html  | tiles:", len(tiles))
