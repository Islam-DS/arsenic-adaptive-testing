"""Do arsenic well-testing studies cite the geostatistical design literature?   python -X utf8 src/scopus_citations.py EXPORT.csv

Input: a Scopus CSV export (with the References field) of
    TITLE-ABS-KEY(arsenic AND (well* OR tubewell*) AND (test* OR screening OR sampling))
retrieved on 24 November 2026 (5,348 records; 5,212 with reference lists for 1995-2026). Scopus exports cannot be
redistributed, so the file is not in this repository; the query above reproduces it (counts can change as Scopus is
updated). Each reference list is searched, case-insensitively, for the title of each design study. The record
identifiers (EID, DOI, year) and the match flags, but no Scopus content, are written to
results/scopus_citation_check.csv, so the check can be verified record by record by anyone with Scopus access.
"""
import pathlib, sys
import pandas as pd

OUT = pathlib.Path(__file__).resolve().parents[1] / "results" / "scopus_citation_check.csv"
TITLES = {
    "Diggle and Lophaven 2006": "bayesian geostatistical design",
    "Chipeta et al. 2016": "adaptive geostatistical design and analysis for prevalence surveys",
    "Kabaghe et al. 2017": "adaptive geostatistical sampling enables efficient identification of malaria hotspots",
}
AUTHORS = {   # looser check by author names, in case a reference title is abbreviated or truncated
    "Diggle and Lophaven 2006": r"diggle.{0,60}lophaven|lophaven.{0,60}diggle",
    "Chipeta et al. 2016": r"chipeta",
    "Kabaghe et al. 2017": r"kabaghe",
}


def main(path):
    d = pd.read_csv(path, low_memory=False)
    d = d[(d.Year >= 1995) & (d.Year <= 2026)]
    refs = d.References.fillna("").str.lower()
    print(f"records 1995-2026: {len(d)}; with reference lists: {(refs != '').sum()}")
    out = pd.DataFrame({"EID": d.EID, "DOI": d.DOI, "Year": d.Year, "has_references": refs != ""})
    for name, title in TITLES.items():
        hit = refs.str.contains(title, regex=False)
        out["cites " + name] = hit
        by_author = refs.str.contains(AUTHORS[name], regex=True)
        out["author match " + name] = by_author
        print(f"{name}: cited by {int(hit.sum())} records (title); author names found in {int(by_author.sum())}")
        for t in d.Title[hit].head(20):
            print("  ", t)
    out.to_csv(OUT, index=False)
    print("written", OUT)


if __name__ == "__main__":
    main(sys.argv[1])
