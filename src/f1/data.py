"""Load and clean the Araihazar well data (HydroShare 8e1373d87419447c945625af13f0a2ea, CC BY 4.0).

Dataset2: ~6,600 wells with lab arsenic (ppb), coordinates, depth (feet) and install year.
Dataset1: 944 wells with field-kit category and lab arsenic, used for the kit error model.
"""
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "araihazar"
SHALLOW_MAX_FT = 100          # wells deeper than this tap the low-arsenic aquifer in this area
BLOCK_M = 2000                # side of the spatial blocks used for the dev/test split
SPLIT_SEED = 20261119
MIN_BLOCK_WELLS = 150


def to_xy(lat, lon, lat0, lon0):
    x = (lon - lon0) * 111_320 * np.cos(np.radians(lat0))
    y = (lat - lat0) * 110_574
    return x, y


def load_wells():
    d = pd.read_csv(RAW / "Dataset2.csv")
    n0 = len(d)
    d = d.dropna(subset=["Latitude", "Longitude"]).copy()
    d["year"] = pd.to_numeric(d["Year"], errors="coerce").where(lambda s: s.between(1950, 2003))
    d["depth_ft"] = d["Depth"]
    d["as_ppb"] = d["As_ppb"].astype(float)
    d["log_as"] = np.log10(d["as_ppb"].clip(lower=1))
    d["deep"] = (d["depth_ft"] > SHALLOW_MAX_FT).astype(int)        # missing depth -> treated as shallow
    d["depth_missing"] = d["depth_ft"].isna().astype(int)
    lat0, lon0 = d["Latitude"].mean(), d["Longitude"].mean()
    d["x"], d["y"] = to_xy(d["Latitude"], d["Longitude"], lat0, lon0)
    d["block"] = (np.floor((d["x"] - d["x"].min()) / BLOCK_M).astype(int).astype(str) + "_" +
                  np.floor((d["y"] - d["y"].min()) / BLOCK_M).astype(int).astype(str))
    # fold blocks with few wells (edge slivers) into the nearest large block, by centroid
    size = d["block"].value_counts()
    cent = d.groupby("block")[["x", "y"]].mean()
    big = size[size >= MIN_BLOCK_WELLS].index
    for b in size[size < MIN_BLOCK_WELLS].index:
        near = ((cent.loc[big] - cent.loc[b]) ** 2).sum(axis=1).idxmin()
        d.loc[d["block"] == b, "block"] = near
    blocks = sorted(d["block"].unique())
    rng = np.random.default_rng(SPLIT_SEED)
    test = set(rng.choice(blocks, size=len(blocks) // 2, replace=False))
    d["split"] = np.where(d["block"].isin(test), "test", "dev")
    d.attrs["dropped_no_coords"] = n0 - len(d)
    d.attrs["origin"] = (lat0, lon0)
    return d.reset_index(drop=True)


def load_kit():
    k = pd.read_csv(RAW / "Dataset1.csv")
    return k.rename(columns={"Kit category": "kit", "Laboratory measured As (ppb)": "lab"})


if __name__ == "__main__":
    d = load_wells()
    print(len(d), "wells; dropped (no coords):", d.attrs["dropped_no_coords"])
    print(d.groupby("split").agg(n=("as_ppb", "size"), blocks=("block", "nunique"),
                                 safe50=("as_ppb", lambda s: (s <= 50).mean()),
                                 deep=("deep", "mean")))
    print(d.groupby(["split", "block"]).size())
