"""Check that every input file is present and matches its documented SHA-256 checksum.   python -X utf8 src/verify_data.py

Exits with status 1 if a required file is missing or differs, so it can stop a pipeline before any run.
The data are not redistributed; see data/README.md for the sources.
"""
import hashlib, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
REQUIRED = {   # file: (sha256, needed for)
    "data/raw/araihazar/Dataset1.csv":
        ("8b16a282f28ff8791ca43f654186ea4c049c15a5c0b4863ecec390df2d172d8b", "944 kit-laboratory pairs"),
    "data/raw/araihazar/Dataset2.csv":
        ("bafe256a83ef7a6ad1cdc0f11c289d7d9db179b6f3ec247cee16a01013a437a9", "2000-01 census"),
    "data/raw/bd_araihazar2012_kit/Well_Water_Arsenic_Concentration.xlsx":
        ("21bd55801c5300b4395ebf192d496369c80eb6fe36f896fc8d3b4155a909eac8", "2012-13 upazila survey"),
}
OPTIONAL = {
    "data/raw/bd_supp_vangeen2019/es9b01375_si_001.xlsx":
        ("6383ccf2d2e63d69fb3afccd949a909c2d5fd35ec4876146d52248f3af5c5a4f", "national village summary (context only)"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    failed = False
    for group, files in (("required", REQUIRED), ("optional", OPTIONAL)):
        for rel, (digest, what) in files.items():
            path = ROOT / rel
            if not path.exists():
                status = "MISSING"
            else:
                status = "OK" if sha256(path) == digest else "CHECKSUM MISMATCH"
            print(f"{status:18s} {group:8s} {rel}  ({what})")
            if group == "required" and status != "OK":
                failed = True
    if failed:
        print("\nRequired input data are missing or differ from the documented versions; see data/README.md.")
        sys.exit(1)
    print("\nAll required input files verified.")


if __name__ == "__main__":
    main()
