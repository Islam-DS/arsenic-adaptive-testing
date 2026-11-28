"""Check that the files in submission/ were generated from the current, audited sources.   python -X utf8 manuscript/check_submission.py

1. appendix.md equals what make_appendix.py produces from the current results;
2. the LaTeX source in the submission zip equals the source generated now from manuscript.md, appendix.md and
   references.bib, and equals manuscript/latex/main.tex;
3. every figure in the zip and in submission/figures is byte-identical to manuscript/figures;
4. the text of submission/manuscript_MDPI_latex.pdf equals the text of a fresh compilation of that source (the version
   line, which carries the build date, is ignored), and every number printed in Table 2 and Tables A1-A10 occurs in it.
Exits with status 1 on any failure.
"""
import hashlib, io, pathlib, re, shutil, subprocess, sys, tempfile, zipfile
import pymupdf

M = pathlib.Path(__file__).resolve().parent
ROOT = M.parent
SUB = ROOT / "submission"
sys.path.insert(0, str(M))
fails = []


def ok(cond, msg):
    print(("OK   " if cond else "FAIL ") + msg)
    if not cond:
        fails.append(msg)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def pdf_text(path):
    t = "".join(p.get_text() for p in pymupdf.open(path))
    return re.sub(r"Version \w+ \d+, \d{4} submitted to", "", re.sub(r"\s+", " ", t))


def main():
    # 1. appendix
    before = (M / "appendix.md").read_text(encoding="utf-8")
    import make_appendix
    make_appendix.main()
    ok((M / "appendix.md").read_text(encoding="utf-8") == before, "appendix.md is up to date with the results")
    # 2. LaTeX source
    import build_latex
    tex, _ = build_latex.make_tex()
    z = zipfile.ZipFile(SUB / "manuscript_MDPI_latex_source.zip")
    ok(z.read("main.tex").decode("utf-8") == tex, "submission zip main.tex = source generated from the current Markdown")
    ok((M / "latex" / "main.tex").read_text(encoding="utf-8") == tex, "manuscript/latex/main.tex = generated source")
    # 3. figures
    figs = {p.name: sha(p.read_bytes()) for p in (M / "figures").glob("*.png")}
    for n in z.namelist():
        if n.startswith("figures/") and n.endswith(".png"):
            ok(figs.get(pathlib.Path(n).name) == sha(z.read(n)), f"zip {n} = manuscript/figures")
    names = {"Figure1.png": "fig1_map.png", "Figure2.png": "fig2_protected.png", "Figure3.png": "fig3_decomposition.png",
             "Figure4.png": "fig4_upazila.png", "Figure5.png": "fig5_robustness.png", "Figure6.png": "fig6_harm.png",
             "FigureA1.png": "figS1_blocks.png", "FigureA2.png": "figA2_calibration.png"}
    for sub, src in names.items():
        p = SUB / "figures" / sub
        ok(p.exists() and sha(p.read_bytes()) == figs.get(src), f"submission/figures/{sub} = manuscript/figures/{src}")
    # 4. PDF: fresh compilation from the zip, text comparison, numbers of the tables
    with tempfile.TemporaryDirectory() as d:
        z.extractall(d)
        for _ in range(2):
            r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=d,
                               capture_output=True, text=True, encoding="utf-8", errors="replace")
        ok(r.returncode == 0, "the submission zip compiles")
        fresh = pdf_text(pathlib.Path(d) / "main.pdf")
    sub_text = pdf_text(SUB / "manuscript_MDPI_latex.pdf")
    ok(sub_text == fresh, "submitted PDF text = fresh compilation of the submitted source")
    md = (M / "manuscript.md").read_text(encoding="utf-8") + (M / "appendix.md").read_text(encoding="utf-8")
    cells = [c.strip() for line in md.split("\n") if line.startswith("|") and not re.match(r"^\|[\s\-|:]+\|$", line)
             for c in line.strip("|").split("|")]
    numbers = {tok for c in cells for tok in re.findall(r"[<+-]?\d[\d,]*\.?\d*", c)}
    missing = sorted(n for n in numbers if n.lstrip("+") not in sub_text and n.replace("-", "−") not in sub_text)
    ok(not missing, f"all {len(numbers)} distinct numbers of the tables occur in the submitted PDF" +
       (f" (missing: {missing[:10]})" if missing else ""))
    if fails:
        sys.exit(1)
    print("\nThe submission files match the current sources.")


if __name__ == "__main__":
    main()
