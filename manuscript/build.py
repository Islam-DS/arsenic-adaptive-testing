"""Build manuscript.docx and manuscript.pdf (MDPI numbered citations) from manuscript.md + references.bib,
with Appendices A-B from appendix.md (written by make_appendix.py).

    python -X utf8 manuscript/build.py
"""
import html, os, pathlib, re, shutil, subprocess, tempfile
import markdown
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, Cm

M = pathlib.Path(__file__).resolve().parent
ROOT = M.parent
CHROME = (os.environ.get("CHROME") or shutil.which("google-chrome") or shutil.which("chromium")
          or shutil.which("chromium-browser") or r"C:\Program Files\Google\Chrome\Application\chrome.exe")
# Chrome/Chromium is used only to print manuscript.pdf from HTML; the MDPI LaTeX version is built by build_latex.py


# ------------------------------------------------------------------ bibliography
def parse_bib(text):
    out = {}
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, re.S):
        kind, key, body = m.groups()
        f = {k.lower(): v.strip() for k, v in re.findall(r"(\w+)\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}", body)}
        f["_kind"] = kind.lower()
        out[key.strip()] = f
    return out


def clean(s):
    s = s.replace("\\&", "&").replace("--", "-").replace("–", "-").replace("—", "-").replace("−", "-")
    return re.sub(r"[{}]", "", s).strip()


def fmt_author(a):
    a = clean(a)
    if "," in a:
        last, first = [x.strip() for x in a.split(",", 1)]
    else:
        parts = a.split()
        last, first = parts[-1], " ".join(parts[:-1])
    initials = "".join(p[0] + "." for p in re.split(r"[\s\-]+", first) if p)
    return f"{last}, {initials}" if initials else last


JOURNAL_ABBR = {'ACS ES&T Water': 'ACS ES&T Water', 'Advances in Water Resources': 'Adv. Water Resour.', 'Applied Water Science': 'Appl. Water Sci.', 'Asian Development Review': 'Asian Dev. Rev.', 'Bulletin of the World Health Organization': 'Bull. World Health Organ.', 'Chemometrics and Intelligent Laboratory Systems': 'Chemom. Intell. Lab. Syst.', 'Environment International': 'Environ. Int.', 'Environment and Development Economics': 'Environ. Dev. Econ.', 'Environmental Geology': 'Environ. Geol.', 'Environmental Health': 'Environ. Health', 'Environmental Health Perspectives': 'Environ. Health Perspect.', 'Environmental Science & Technology': 'Environ. Sci. Technol.', 'GeoHealth': 'GeoHealth', 'Geoscience Frontiers': 'Geosci. Front.', 'Global Environmental Change': 'Glob. Environ. Change', 'Health & Place': 'Health Place', 'Journal of Artificial Intelligence Research': 'J. Artif. Intell. Res.', 'IEEE Transactions on Information Theory': 'IEEE Trans. Inf. Theory', 'JAMA': 'JAMA', 'Journal of Contaminant Hydrology': 'J. Contam. Hydrol.', 'Journal of Development Economics': 'J. Dev. Econ.', 'Journal of Environmental Economics and Management': 'J. Environ. Econ. Manag.', 'Journal of Environmental Science and Health, Part A': 'J. Environ. Sci. Health A', 'Journal of Public Health Management & Practice': 'J. Public Health Manag. Pract.', 'Journal of Public Health Management and Practice': 'J. Public Health Manag. Pract.', 'Journal of Water, Sanitation and Hygiene for Development': 'J. Water Sanit. Hyg. Dev.', 'Journal of the American Statistical Association': 'J. Am. Stat. Assoc.', 'Nature Reviews Earth & Environment': 'Nat. Rev. Earth Environ.', 'Nature Sustainability': 'Nat. Sustain.', 'PLOS Neglected Tropical Diseases': 'PLoS Negl. Trop. Dis.', 'PLOS ONE': 'PLoS ONE', 'PNAS Nexus': 'PNAS Nexus', 'Proceedings of the IEEE': 'Proc. IEEE', 'Risk Analysis': 'Risk Anal.', 'Scandinavian Journal of Statistics': 'Scand. J. Stat.', 'Science': 'Science', 'Science of The Total Environment': 'Sci. Total Environ.', 'Scientific Reports': 'Sci. Rep.', 'Spatial Statistics': 'Spat. Stat.', 'Sustainability': 'Sustainability', 'American Journal of Tropical Medicine and Hygiene': 'Am. J. Trop. Med. Hyg.', 'The Lancet': 'Lancet', 'The World Bank Economic Review': 'World Bank Econ. Rev.', 'Water': 'Water', 'Water International': 'Water Int.', 'Water Research': 'Water Res.', 'Water Resources Research': 'Water Resour. Res.'}


def fmt_ref(e):
    authors = [fmt_author(a) for a in e.get("author", "").split(" and ") if a.strip()]
    if len(authors) > 10:
        authors = authors[:10] + ["et al."]
    au = "; ".join(authors)
    title = clean(e.get("title", ""))
    year = e.get("year", "")
    doi = e.get("doi", "")
    if e["_kind"] == "article" and e.get("journal"):
        s = f"{au} {title}. *{JOURNAL_ABBR.get(clean(e['journal']), clean(e['journal']))}* **{year}**"
        if e.get("volume"):
            s += f", *{e['volume']}*"
        if e.get("pages"):
            s += f", {clean(e['pages'])}"
        s += "."
    else:
        where = clean(e.get("journal") or e.get("booktitle") or e.get("howpublished") or e.get("publisher") or "")
        s = f"{au} {title}. {where + ', ' if where else ''}{year}."
    if doi:
        s += f" https://doi.org/{doi}"
    return s


def number_citations(md, bib):
    order = []
    def repl(m):
        keys = re.findall(r"@([A-Za-z][A-Za-z0-9]+)", m.group(0))
        nums = []
        for k in keys:
            if k not in bib:
                raise KeyError(k)
            if k not in order:
                order.append(k)
            nums.append(order.index(k) + 1)
        nums = sorted(set(nums))
        parts, i = [], 0
        while i < len(nums):          # compress runs: 3,4,5 -> 3–5
            j = i
            while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                j += 1
            parts.append(str(nums[i]) if j == i else (f"{nums[i]},{nums[j]}" if j == i + 1 else f"{nums[i]}-{nums[j]}"))
            i = j + 1
        return "[" + ",".join(parts) + "]"
    body = re.sub(r"\[@[^\]]+\]", repl, md)
    refs = [fmt_ref(bib[k]) for k in order]
    return body, refs


# ------------------------------------------------------------------ figures / tables numbering
def number_figures(md):
    n = [0]
    def repl(m):
        if m.group(1).startswith("**Figure"):       # appendix figures carry their own label (Figure A1)
            return m.group(0)
        n[0] += 1
        return f"![**Figure {n[0]}.** {m.group(1)}]({m.group(2)})"
    return re.sub(r"!\[(.*?)\]\((.*?)\)", repl, md)


# ------------------------------------------------------------------ docx writer
INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|\$[^$]+\$)")


def add_runs(p, text, size=None):
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**"):
            r = p.add_run(part[2:-2]); r.bold = True
        elif part.startswith("*"):
            r = p.add_run(part[1:-1]); r.italic = True
        elif part.startswith("$"):
            r = p.add_run(latex_to_text(part[1:-1])); r.italic = True
        else:
            r = p.add_run(part.replace(DOLLAR, "$"))
        if size:
            r.font.size = Pt(size)


DOLLAR = "§DOLLAR§"
AUTHOR_BLOCK = ["1 Al-Farabi Kazakh National University, Almaty, Kazakhstan; islam_md_mahbubul@live.kaznu.kz (M.M.I.); yerkos.ainur@kaznu.kz (A.Y.)",
                "2 Riga Technical University, Riga, Latvia; Nadezda.Kunicina@rtu.lv (N.K.)",
                "* Correspondence: islam_md_mahbubul@live.kaznu.kz (M.M.I.); Nadezda.Kunicina@rtu.lv (N.K.)",
                "ORCID: M.M.I. 0009-0009-1910-1980; A.Y. 0000-0001-5949-6942; N.K. 0000-0002-0980-0958"]


def latex_to_text(s):
    for a, b in (("\\lambda", "λ"), ("\\kappa", "κ"), ("\\bar", ""), ("\\sigma", "σ"), ("\\tau", "τ"), ("\\rho", "ρ"), ("\\mu", "μ"), ("\\Phi", "Φ"),
                 ("\\sum", "Σ"), ("\\prod", "Π"), ("\\neq", " ≠ "), ("\\in", " ∈ "), ("\\sqrt", "√"), ("\\exp", "exp"),
                 ("\\log", "log"), ("\\max", "max"), ("^{\\top}", "ᵀ"), ("\\top", "ᵀ"), ("\\mathbf", ""),
                 ("\\mathrm", ""), ("\\,", " "), ("\\text", "")):
        s = s.replace(a, b)
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"\\frac(\S+)", r"\1", s)
    return s.replace("_", "").replace("\\", "")


AUTHORS_META = "Md. Mahbubul Islam; Ainur Yerkos; Nadezhda Kunicina"


def _props(doc, title):
    cp = doc.core_properties
    cp.author = AUTHORS_META
    cp.last_modified_by = "Md. Mahbubul Islam"
    cp.title = title
    cp.comments = ""
    cp.keywords = ""
    cp.subject = ""


def write_docx(md, refs, path, title, author):
    doc = Document()
    st = doc.styles["Normal"]; st.font.name = "Palatino Linotype"; st.font.size = Pt(10)
    for s in doc.sections:
        s.left_margin = s.right_margin = Cm(2.2)
    h = doc.add_paragraph(); r = h.add_run("Article"); r.italic = True
    p = doc.add_paragraph(); r = p.add_run(title); r.bold = True; r.font.size = Pt(16)
    p = doc.add_paragraph(); r = p.add_run(author); r.bold = True
    for line in AUTHOR_BLOCK:
        q = doc.add_paragraph(); rr = q.add_run(line); rr.font.size = Pt(9)
    lines = md.split("\n")
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("# "):
            doc.add_heading(ln[2:].strip(), level=1)
        elif ln.startswith("## "):
            doc.add_heading(ln[3:].strip(), level=2)
        elif ln.startswith("!["):
            m = re.match(r"!\[(.*)\]\((.*)\)", ln)
            doc.add_picture(str(M / m.group(2)), width=Cm(16))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_runs(doc.add_paragraph(), m.group(1), size=9)
        elif ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                if not re.match(r"^\|[\s\-|:]+\|$", lines[i]):
                    rows.append([c.strip() for c in lines[i].strip("|").split("|")])
                i += 1
            t = doc.add_table(rows=len(rows), cols=len(rows[0])); t.style = "Light List"
            for a, row in enumerate(rows):
                for b, c in enumerate(row):
                    cell = t.cell(a, b); cell.text = ""
                    add_runs(cell.paragraphs[0], c, size=8.5)
            continue
        elif ln.startswith("$$"):
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run("gⱼ = [ pⱼ (Eⱼ + λUⱼ) + (1 - pⱼ)(Sⱼ + λQⱼ) ] / ( t_test + t_walk(j) )"); r.italic = True
        elif ln.startswith("<!--"):
            pass
        elif ln.startswith("- "):
            add_runs(doc.add_paragraph(style="List Bullet"), ln[2:])
        elif ln.strip():
            add_runs(doc.add_paragraph(), ln)
        i += 1
    for n, ref in enumerate(refs, 1):
        p = doc.add_paragraph(); add_runs(p, f"{n}. {ref}", size=9)
    _props(doc, title)
    doc.save(path)


# ------------------------------------------------------------------ html/pdf for reading
CSS = """@page{size:A4;margin:18mm 17mm} body{font-family:'Palatino Linotype',Georgia,serif;font-size:10.5pt;line-height:1.45;color:#111}
h1{font-size:13pt;margin:16pt 0 6pt} h2{font-size:11pt;font-style:italic;margin:12pt 0 4pt} .title{font-size:17pt;font-weight:bold;margin:4pt 0}
img{max-width:100%;display:block;margin:8pt auto 2pt} p{text-align:justify;margin:4pt 0} table{border-collapse:collapse;font-size:8.5pt;margin:6pt auto}
th,td{border-top:0.6pt solid #888;border-bottom:0.6pt solid #888;padding:2pt 5pt} .ref{font-size:9pt;margin:2pt 0} figure{margin:0}"""


def write_pdf(md, refs, title, author, pdf):
    body = re.sub(r"\$\$.*?\$\$", lambda m: "<p style='text-align:center'><i>g<sub>j</sub></i> = [ <i>p<sub>j</sub></i> (<i>E<sub>j</sub></i> + λ<i>U<sub>j</sub></i>) + (1 - <i>p<sub>j</sub></i>)(<i>S<sub>j</sub></i> + λ<i>Q<sub>j</sub></i>) ] / ( <i>t</i><sub>test</sub> + <i>t</i><sub>walk</sub>(<i>j</i>) ),</p>", md, flags=re.S)
    body = re.sub(r"\$([^$]+)\$", lambda m: "<i>" + latex_to_text(m.group(1)) + "</i>", body)
    body = body.replace(DOLLAR, "$")
    h = markdown.markdown(body, extensions=["tables"])
    h = re.sub(r'<img alt="(.*?)" src="(.*?)" />',
               lambda m: f'<img src="{(M / m.group(2)).as_uri()}"><p style="font-size:9pt">'
                         f'{markdown.markdown(html.unescape(m.group(1)))[3:-4]}</p>', h)
    rf = "".join(f'<p class="ref">{n}. {markdown.markdown(r)[3:-4]}</p>' for n, r in enumerate(refs, 1))
    doc = (f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>"
           f"<p><i>Article</i></p><p class='title'>{title}</p><p><b>{author}</b></p>"
           + "".join(f"<p style='font-size:9pt;margin:1pt 0'>{x}</p>" for x in AUTHOR_BLOCK) + f"{h}{rf}</body></html>")
    tmp = M / "manuscript.html"
    tmp.write_text(doc, encoding="utf-8")
    with tempfile.TemporaryDirectory() as prof:
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", f"--user-data-dir={prof}", "--no-pdf-header-footer",
                        f"--print-to-pdf={pdf}", tmp.as_uri()], check=True, timeout=120, capture_output=True)
    import pymupdf
    d = pymupdf.open(pdf)
    d.set_metadata({"title": title, "author": AUTHORS_META, "creator": "", "producer": "", "subject": "", "keywords": ""})
    d.save(str(pdf) + ".tmp"); d.close(); pathlib.Path(str(pdf) + ".tmp").replace(pdf)


if __name__ == "__main__":
    src = (M / "manuscript.md").read_text(encoding="utf-8")
    fm = re.match(r"---\n(.*?)\n---\n", src, re.S)
    meta = dict(re.findall(r'(\w+): "?(.*?)"?$', fm.group(1), re.M))
    md = src[fm.end():]
    app = M / "appendix.md"
    if app.exists():                                  # Appendices A-B go after the back matter, before the references
        md = md.replace("\n# References", "\n" + app.read_text(encoding="utf-8") + "\n\n# References")
    bib = parse_bib((M / "references.bib").read_text(encoding="utf-8"))
    md = number_figures(md).replace("US$", "US" + DOLLAR)   # keep currency out of the $math$ regex
    md, refs = number_citations(md, bib)
    write_docx(md, refs, M / "manuscript.docx", meta["title"], meta["author"])
    write_pdf(md, refs, meta["title"], meta["author"], M / "manuscript.pdf")
    print(len(refs), "references; built manuscript.docx and manuscript.pdf (appendices included)")
