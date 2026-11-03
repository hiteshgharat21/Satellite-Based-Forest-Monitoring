
import ee

GEE_PROJECT = "uk-forest-ai"   
START_YEAR, END_YEAR = 2019, 2025
TILE_DEG = 0.1                        
MIN_FOREST_FRAC = 0.20                
SCALE = 100                           
DRIVE_FOLDER = "uk_forest"

ee.Initialize(project=GEE_PROJECT)

# ---------- 1. Uttarakhand boundary ----------
uk = (ee.FeatureCollection("FAO/GAUL/2015/level1")
      .filter(ee.Filter.eq("ADM1_NAME", "Uttarakhand")))
print("Uttarakhand features found:", uk.size().getInfo())   
uk_geom = uk.geometry()

# ---------- 2. Forest mask (ESA WorldCover, class 10 = Tree cover) ----------
worldcover = ee.Image("ESA/WorldCover/v200/2021").select("Map")
forest = worldcover.eq(10).rename("forest")

# ---------- 3. Grid of tiles ----------
bounds = uk_geom.bounds()
coords = ee.List(bounds.coordinates().get(0))
xmin = ee.Number(ee.List(coords.get(0)).get(0)); ymin = ee.Number(ee.List(coords.get(0)).get(1))
xmax = ee.Number(ee.List(coords.get(2)).get(0)); ymax = ee.Number(ee.List(coords.get(2)).get(1))

xs = ee.List.sequence(xmin, xmax, TILE_DEG)
ys = ee.List.sequence(ymin, ymax, TILE_DEG)

def make_row(x):
    x = ee.Number(x)
    def make_cell(y):
        y = ee.Number(y)
        return ee.Feature(ee.Geometry.Rectangle([x, y, x.add(TILE_DEG), y.add(TILE_DEG)]),
                          {"cx": x.add(TILE_DEG / 2), "cy": y.add(TILE_DEG / 2)})
    return ys.map(make_cell)

grid = ee.FeatureCollection(xs.map(make_row).flatten()).filterBounds(uk_geom)


grid = forest.reduceRegions(collection=grid, reducer=ee.Reducer.mean(), scale=SCALE)
grid = (grid.filter(ee.Filter.gte("mean", MIN_FOREST_FRAC))
            .map(lambda f: f.set({
                "forest_frac": f.get("mean"),
                "tile_id": ee.String("UK_").cat(ee.Number(f.get("cx")).multiply(100).round().format("%d"))
                              .cat("_").cat(ee.Number(f.get("cy")).multiply(100).round().format("%d"))})))
print("Forest tiles:", grid.size().getInfo())

# ---------- 4. Static terrain ----------
dem = ee.Image("USGS/SRTMGL1_003")
terrain = ee.Terrain.products(dem).select(["elevation", "slope"])

# ---------- 5. Monthly image builder ----------
def mask_s2(img):
    scl = img.select("SCL")
    
    good = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    return img.updateMask(good).divide(10000).copyProperties(img, ["system:time_start"])

def monthly_image(year, month):
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")

    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(uk_geom).filterDate(start, end)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
          .map(mask_s2).median())
    ndvi = s2.normalizedDifference(["B8", "B4"]).rename("ndvi")
    ndmi = s2.normalizedDifference(["B8", "B11"]).rename("ndmi")
    nbr = s2.normalizedDifference(["B8", "B12"]).rename("nbr")
    # sirf jungle pixels ka average lo
    idx = ndvi.addBands(ndmi).addBands(nbr).updateMask(forest)

    era = (ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
           .filterDate(start, end).first())
    temp = era.select("temperature_2m").subtract(273.15).rename("temp_c")
    precip = era.select("total_precipitation_sum").multiply(1000).rename("precip_mm")
    soil = era.select("volumetric_soil_water_layer_1").rename("soil_moist")

    burn = (ee.ImageCollection("MODIS/061/MCD64A1").filterDate(start, end).first()
            .select("BurnDate").gt(0).unmask(0).rename("burned_frac"))

    return (idx.addBands(temp).addBands(precip).addBands(soil)
               .addBands(terrain).addBands(burn))

def tiles_for_month(year, month):
    img = monthly_image(year, month)
    res = img.reduceRegions(collection=grid, reducer=ee.Reducer.mean(), scale=SCALE, tileScale=4)
    return res.map(lambda f: f.set({"year": year, "month": month}).setGeometry(None))


COLS = ["tile_id", "cx", "cy", "year", "month", "ndvi", "ndmi", "nbr", "temp_c",
        "precip_mm", "soil_moist", "elevation", "slope", "forest_frac", "burned_frac"]

for year in range(START_YEAR, END_YEAR + 1):
    fc = ee.FeatureCollection([tiles_for_month(year, m) for m in range(1, 13)]).flatten()
    task = ee.batch.Export.table.toDrive(
        collection=fc, description=f"uk_features_{year}", folder=DRIVE_FOLDER,
        fileNamePrefix=f"uk_features_{year}", fileFormat="CSV", selectors=COLS)
    task.start()
    print("Started export:", year)

print(" https://code.earthengine.google.com/tasks ")
