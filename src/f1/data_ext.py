"""Araihazar upazila 2012-13 field-kit survey (HydroShare 48e779d1, CC BY 4.0) and its link to the 2000-01 census."""
import numpy as np
import pandas as pd
from data import ROOT, SHALLOW_MAX_FT

MIN_UNIT = 150
SEED = 20261120


# ------------------------------------------------------------------ Araihazar upazila 2012-13 blanket survey
BD2012 = ROOT / "data" / "raw" / "bd_araihazar2012_kit" / "Well_Water_Arsenic_Concentration.xlsx"


def load_bd2012(block_m=2000):
    """46,976 wells (HydroShare 48e779d1, CC BY 4.0): field-kit arsenic (ITS Econo-Quick), GPS, owner-reported depth (m).
    Kit readings are the truth of the replay (no extra kit error). Campaign units: 2 km blocks (merged if < 150 wells),
    split at random into halves for cross-fitting, as for the 2000-01 census."""
    d = pd.read_excel(BD2012, sheet_name="Dissolved As Concentration")
    d.columns = [str(c).strip() for c in d.columns]
    d["as_ppb"] = pd.to_numeric(d["As_Result"], errors="coerce")
    d = d.dropna(subset=["as_ppb", "lat", "lon"]).copy()
    d["log_as"] = np.log10(d["as_ppb"].clip(lower=1))
    dep = pd.to_numeric(d["Depth_m"], errors="coerce")
    d["depth_ft"] = (dep * 3.281).where(dep > 0)
    d["deep"] = (d["depth_ft"] > SHALLOW_MAX_FT).astype(int)
    d["depth_missing"] = d["depth_ft"].isna().astype(int)
    # link to the 2000-01 laboratory census by well ID (earlier survey for a re-testing scenario)
    old = pd.read_csv(ROOT / "data" / "raw" / "araihazar" / "Dataset2.csv")[["Well_ID", "As_ppb", "Latitude", "Longitude"]]
    old = old.drop_duplicates("Well_ID").set_index("Well_ID")
    wid = pd.to_numeric(d["Well ID"], errors="coerce")
    d["old_as"] = wid.map(old["As_ppb"]).where(wid > 0)
    # spatial check: keep a link only if the two recorded positions are within 100 m
    olat, olon = wid.map(old["Latitude"]), wid.map(old["Longitude"])
    dist = np.sqrt(((d["lon"] - olon) * 111_320 * np.cos(np.radians(d["lat"]))) ** 2 + ((d["lat"] - olat) * 110_574) ** 2)
    d["old_dist_m"] = dist
    d.loc[~(dist <= 100), "old_as"] = np.nan
    lat0, lon0 = d["lat"].mean(), d["lon"].mean()
    d["x"] = (d["lon"] - lon0) * 111_320 * np.cos(np.radians(lat0))
    d["y"] = (d["lat"] - lat0) * 110_574
    d["block"] = (np.floor((d["x"] - d["x"].min()) / block_m).astype(int).astype(str) + "_" +
                  np.floor((d["y"] - d["y"].min()) / block_m).astype(int).astype(str))
    size = d["block"].value_counts()
    cent = d.groupby("block")[["x", "y"]].mean()
    big = size[size >= MIN_UNIT].index
    for b in size[size < MIN_UNIT].index:
        near = ((cent.loc[big] - cent.loc[b]) ** 2).sum(axis=1).idxmin()
        d.loc[d["block"] == b, "block"] = near
    blocks = sorted(d["block"].unique())
    rng = np.random.default_rng(SEED)
    half = set(rng.choice(blocks, size=len(blocks) // 2, replace=False))
    d["split"] = np.where(d["block"].isin(half), "test", "dev")
    return d.reset_index(drop=True)


if __name__ == "__main__":
    from scipy.spatial import cKDTree
    for a in ["bd2012"]:
        d = load_bd2012()
        X = d[["x", "y"]].to_numpy()
        nn = np.median([len(v) - 1 for v in cKDTree(X).query_ball_point(X, 100)])
        print(f"{a}: {len(d)} wells, {d.block.nunique()} units (sizes {d.block.value_counts().min()}–"
              f"{d.block.value_counts().max()}), median nbrs<100m {nn:.0f}, safe50 {(d.as_ppb<=50).mean():.2f}, "
              f"safe10 {(d.as_ppb<=10).mean():.2f}, depth {1-d.depth_missing.mean():.2f}, deep {d.deep.mean():.2f}")
