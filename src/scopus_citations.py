"""Do arsenic well-testing studies cite the geostatistical design literature?   python -X utf8 src/scopus_citations.py EXPORT.csv

Input: a Scopus CSV export (with the References field) of
    TITLE-ABS-KEY(arsenic AND (well* OR tubewell*) AND (test* OR screening OR sampling))
retrieved on 24 November 2026 (5,348 records; 5,212 with reference lists for 1995-2026). Scopus exports cannot be
redistributed, so the file is not in this repository; the query above reproduces it (counts can change as Scopus is
updated). Each reference list is searched, case-insensitively, for the title of each design study.
"""
import sys
import pandas as pd

TITLES = {
    "Diggle and Lophaven 2006": "bayesian geostatistical design",
    "Chipeta et al. 2016": "adaptive geostatistical design and analysis for prevalence surveys",
    "Kabaghe et al. 2017": "adaptive geostatistical sampling enables efficient identification of malaria hotspots",
}


def main(path):
    d = pd.read_csv(path, low_memory=False)
    d = d[(d.Year >= 1995) & (d.Year <= 2026)]
    refs = d.References.fillna("").str.lower()
    print(f"records 1995-2026: {len(d)}; with reference lists: {(refs != '').sum()}")
    for name, title in TITLES.items():
        hits = d[refs.str.contains(title, regex=False)]
        print(f"{name}: cited by {len(hits)} records" + ("".join(f"\n  {t}" for t in hits.Title.head(20)) if len(hits) else ""))


if __name__ == "__main__":
    main(sys.argv[1])
