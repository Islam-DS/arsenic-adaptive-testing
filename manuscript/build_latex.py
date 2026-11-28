"""Convert manuscript.md + references.bib into the MDPI Sustainability LaTeX template and compile it.

    python -X utf8 manuscript/build_latex.py

Output: manuscript/latex/main.tex and main.pdf (class files in manuscript/latex/Definitions, see SOURCE.md there).
In-text citations link to the reference list (natbib + hyperref from mdpi.cls); every reference has a clickable DOI.
"""
import pathlib, re, subprocess, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from build import parse_bib, clean, fmt_author, JOURNAL_ABBR

M = pathlib.Path(__file__).resolve().parent
L = M / "latex"
ORCID = "0009-0009-1910-1980"
AFFIL = "Al-Farabi Kazakh National University, Almaty, Kazakhstan"
EMAIL = "islam_md_mahbubul@live.kaznu.kz"

UNI = {"µ": r"\textmu{}", "×": r"$\times$", "−": r"$-$", "≤": r"$\leq$", "≥": r"$\geq$", "²": r"\textsuperscript{2}",
       "₁₀": r"$_{10}$", "λ": r"$\lambda$", "½": r"1/2", "’": "'", "‘": "`", "“": "``", "”": "''", "…": r"\ldots{}", "∗": r"$\ast$", "é": r"\'{e}"}


def esc_text(s):
    """Escape a plain-text fragment (no math inside)."""
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in (("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"), ("$", r"\$"), ("~", r"\textasciitilde{}")):
        s = s.replace(a, b)
    for a, b in UNI.items():
        s = s.replace(a, b)
    return s


def inline(s):
    """Markdown inline -> LaTeX: math kept, citations, bold, italics, escaping."""
    s = s.replace("US$", "US\x01")                       # currency, not math
    out, pos = [], 0
    for m in re.finditer(r"\$[^$]+\$", s):                 # keep inline math untouched
        out.append(("t", s[pos:m.start()])); out.append(("m", m.group(0))); pos = m.end()
    out.append(("t", s[pos:]))
    res = []
    for kind, frag in out:
        if kind == "m":
            res.append(frag.replace("\\text", "\\mathrm"))
            continue
        cites = []
        frag = re.sub(r"\[@([^\]]+)\]", lambda m: (cites.append(",".join(k.strip().lstrip("@") for k in m.group(1).split(";"))),
                                                    f"\x00{len(cites)-1}\x00")[1], frag)
        frag = esc_text(frag).replace("\x01", r"\$")
        frag = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", frag)
        frag = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\textit{\1}", frag)
        frag = re.sub(r"\x00(\d+)\x00", lambda m: r"~\cite{" + cites[int(m.group(1))] + "}", frag)
        res.append(frag)
    return "".join(res)


def ref_tex(e):
    authors = [fmt_author(a) for a in e.get("author", "").split(" and ") if a.strip()]
    if len(authors) > 10:
        authors = authors[:10] + ["et al."]
    au = esc_text("; ".join(authors))
    title = esc_text(clean(e.get("title", "")))
    year = e.get("year", "")
    if e["_kind"] == "article" and e.get("journal"):
        j = JOURNAL_ABBR.get(clean(e["journal"]), clean(e["journal"]))
        s = f"{au} {title}. \\textit{{{esc_text(j)}}} \\textbf{{{year}}}"
        if e.get("volume"):
            s += f", \\textit{{{e['volume']}}}"
        if e.get("pages"):
            s += f", {esc_text(clean(e['pages']))}"
        s += "."
    else:
        where = clean(e.get("journal") or e.get("booktitle") or e.get("howpublished") or e.get("publisher") or "")
        s = f"{au} {title}. {esc_text(where) + ', ' if where else ''}{year}."
    if e.get("doi"):
        s += f" \\url{{https://doi.org/{e['doi']}}}"
    return s


def body_to_tex(md, label_prefix=""):
    lines, out, i = md.split("\n"), [], 0
    fig_n = tab_n = 0
    spec_hint = None
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("<!-- spec:"):
            spec_hint = ln[len("<!-- spec:"):].rstrip("->").strip()
            i += 1
            continue
        if ln.startswith("# "):
            title = re.sub(r"^#\s+(\d+\.|Appendix [A-Z]\.)\s*", "", ln).strip()
            out.append(f"\\section{{{inline(title)}}}")
        elif ln.startswith("## "):
            title = re.sub(r"^##\s+(\d+\.\d+\.\s*)?", "", ln).strip()
            out.append(f"\\subsection{{{inline(title)}}}")
        elif ln.startswith("!["):
            m = re.match(r"!\[(.*)\]\((.*)\)", ln)
            fig_n += 1
            cap = re.sub(r"^\*\*Figure [A-Z]?\d+\.\*\*\s*", "", m.group(1))
            out.append("\\begin{figure}[!htbp]\n\\centering\n"
                       f"\\includegraphics[width=\\textwidth]{{{m.group(2)}}}\n"
                       f"\\caption{{{inline(cap)}}}\n\\label{{{label_prefix}fig{fig_n}}}\n\\end{{figure}}")
        elif ln.startswith("**Table"):
            cap = re.sub(r"^\*\*Table [A-Z]?\d+\.\*\*\s*", "", ln)
            j = i + 1
            while j < len(lines) and not lines[j].startswith("|"):
                j += 1
            rows = []
            while j < len(lines) and lines[j].startswith("|"):
                if not re.match(r"^\|[\s\-|:]+\|$", lines[j]):
                    rows.append([c.strip() for c in lines[j].strip("|").split("|")])
                j += 1
            tab_n += 1
            ncol = len(rows[0])
            if spec_hint:
                spec, size = spec_hint, "\\scriptsize\\setlength{\\tabcolsep}{3pt}"
            else:
                spec, size = ("p{3.4cm}p{5.2cm}p{5.2cm}" if ncol == 3 else "p{2.5cm}p{2.6cm}" + "c" * (ncol - 2)), "\\footnotesize"
            spec_hint = None
            head = " & ".join(f"\\textbf{{{inline(c)}}}" for c in rows[0]) + " \\\\"
            data = [" & ".join(inline(c) for c in r) + " \\\\" for r in rows[1:]]
            label = f"{label_prefix}tab{tab_n}"
            if len(data) > 40:                       # long table: breaks across pages, full width incl. left margin
                out.append("{" + size + "\\setlength{\\tabcolsep}{3pt}\\setlength{\\LTleft}{-\\extralength}"
                           "\\setlength{\\LTright}{0pt plus 1fill}\\setlength{\\LTcapwidth}{\\textwidth}\n"
                           f"\\begin{{longtable}}{{{spec}}}\n\\caption{{{inline(cap)}}}\\label{{{label}}}\\\\\n"
                           f"\\toprule\n{head}\n\\midrule\n\\endfirsthead\n"
                           f"\\multicolumn{{{ncol}}}{{l}}{{\\textit{{Table \\thetable{{}} (continued)}}}}\\\\\n\\toprule\n{head}\n\\midrule\n\\endhead\n"
                           "\\bottomrule\n\\endlastfoot\n" + "\n".join(data) + "\n\\end{longtable}}")
            else:
                place = "H" if label_prefix else "!htbp"     # appendix tables stay in order, before the references
                out.append(f"\\begin{{table}}[{place}]\n" f"\\caption{{{inline(cap)}}}\n\\label{{{label}}}\n"
                           f"{size}\n\\begin{{adjustwidth}}{{-\\extralength}}{{0cm}}\\centering\n"
                           f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{head}\n\\midrule\n" + "\n".join(data) +
                           "\n\\bottomrule\n\\end{tabular}\n\\end{adjustwidth}\n\\end{table}")
            i = j
            continue
        elif ln.startswith("**Algorithm"):
            cap = inline(ln)
            j = i + 1
            while j < len(lines) and not re.match(r"^\d+\. ", lines[j]):
                j += 1
            items = []
            while j < len(lines) and re.match(r"^\d+\. ", lines[j]):
                items.append(r"\item " + inline(re.sub(r"^\d+\.\s*", "", lines[j])))
                j += 1
            out.append("\\begin{figure}[H]\n\\noindent\\fbox{\\begin{minipage}{0.96\\linewidth}\\small\n" + cap +
                       "\n\\begin{enumerate}[leftmargin=*,labelsep=2mm,itemsep=0pt]\n" + "\n".join(items) +
                       "\n\\end{enumerate}\n\\end{minipage}}\n\\end{figure}")
            i = j
            continue
        elif ln.startswith("$$"):
            eq = ln.strip().strip("$").strip().rstrip(",").strip()
            out.append("\n\\begin{equation}\n" + eq.replace("\\text", "\\mathrm") + ",\n\\end{equation}\n")
        elif ln.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append("\\item " + inline(lines[i][2:]))
                i += 1
            out.append("\\begin{itemize}[leftmargin=*,labelsep=5.8mm]\n" + "\n".join(items) + "\n\\end{itemize}")
            continue
        elif ln.strip():
            out.append(inline(ln))
        else:
            out.append("")
        i += 1
    return "\n".join(out)


def section_text(md, name):
    m = re.search(rf"^# {re.escape(name)}\n(.*?)(?=^# |\Z)", md, re.S | re.M)
    return m.group(1).strip() if m else ""


def make_tex():
    """The LaTeX source generated from manuscript.md, appendix.md and references.bib."""
    src = (M / "manuscript.md").read_text(encoding="utf-8")
    fm = re.match(r"---\n(.*?)\n---\n", src, re.S)
    meta = dict(re.findall(r'(\w+): "?(.*?)"?$', fm.group(1), re.M))
    md = src[fm.end():]
    appendix_md = (M / "appendix.md").read_text(encoding="utf-8") if (M / "appendix.md").exists() else ""
    bib = parse_bib((M / "references.bib").read_text(encoding="utf-8"))
    order = []
    for m in re.finditer(r"\[@([^\]]+)\]", md + appendix_md):
        for k in m.group(1).split(";"):
            k = k.strip().lstrip("@")
            if k not in order:
                order.append(k)
    missing = [k for k in order if k not in bib]
    assert not missing, missing

    abstract = section_text(md, "Abstract")
    kw = re.search(r"\*\*Keywords:\*\*\s*(.*)", abstract).group(1)
    abstract = abstract.split("**Keywords:**")[0].strip()
    b0 = md.index("# 1. Introduction"); b1 = md.index("# Author Contributions")
    body = body_to_tex(md[b0:b1].strip())
    appendix = body_to_tex(appendix_md.strip(), label_prefix="app") if appendix_md else ""
    back = {k: inline(section_text(md, v)) for k, v in (
        ("supp", "Supplementary Materials"), ("contrib", "Author Contributions"), ("funding", "Funding"),
        ("irb", "Institutional Review Board Statement"), ("consent", "Informed Consent Statement"),
        ("data", "Data Availability Statement"), ("ack", "Acknowledgments"), ("coi", "Conflicts of Interest"))}
    abbr = re.findall(r"^- (.+?): (.+)$", section_text(md, "Abbreviations"), re.M)
    abbr_tex = "\n".join(f"{esc_text(a)} & {esc_text(b)}\\\\" for a, b in abbr)
    refs = "\n".join(f"\\bibitem{{{k}}}\n{ref_tex(bib[k])}\n" for k in order)
    title = meta["title"]

    supp_tex = ("\\supplementary{" + back["supp"] + "}\n\n") if back["supp"] else ""
    app_tex = ("\\appendixtitles{yes}\n\\appendixstart\n\\appendix\n" + appendix + "\n") if appendix else ""
    tex = rf"""\documentclass[sustainability,article,submit,pdftex,moreauthors]{{Definitions/mdpi}}
\usepackage{{longtable}}
\newcolumntype{{L}}[1]{{>{{\raggedright\arraybackslash}}p{{#1}}}}
\newcolumntype{{R}}[1]{{>{{\raggedleft\arraybackslash}}p{{#1}}}}
\makeatletter\hyperrefpackage
\patchcmd{{\NAT@citexnum}}{{--\NAT@penalty}}{{-\NAT@penalty}}{{}}{{}}
\makeatother

\firstpage{{1}}
\makeatletter
\setcounter{{page}}{{\@firstpage}}
\makeatother
\pubvolume{{1}}
\issuenum{{1}}
\articlenumber{{0}}
\pubyear{{2026}}
\copyrightyear{{2026}}
\datereceived{{ }}
\daterevised{{ }}
\dateaccepted{{ }}
\datepublished{{ }}
\hreflink{{https://doi.org/}}

\Title{{{esc_text(title)}}}
\TitleCitation{{{esc_text(title)}}}
\newcommand{{\orcidauthorA}}{{0009-0009-1910-1980}}
\newcommand{{\orcidauthorB}}{{0000-0001-5949-6942}}
\newcommand{{\orcidauthorC}}{{0000-0002-0980-0958}}
\Author{{Md. Mahbubul Islam $^{{1,}}$*\orcidA{{}}, Ainur Yerkos $^{{1}}$\orcidB{{}} and Nadezhda Kunicina $^{{2,}}$*\orcidC{{}}}}
\AuthorNames{{Md. Mahbubul Islam, Ainur Yerkos and Nadezhda Kunicina}}
\AuthorCitation{{Islam, M.M.; Yerkos, A.; Kunicina, N.}}
\address{{%
$^{{1}}$ \quad Al-Farabi Kazakh National University, Almaty, Kazakhstan; \href{{mailto:islam_md_mahbubul@live.kaznu.kz}}{{islam\_md\_mahbubul@live.kaznu.kz}} (M.M.I.); \href{{mailto:yerkos.ainur@kaznu.kz}}{{yerkos.ainur@kaznu.kz}} (A.Y.)\\
$^{{2}}$ \quad Riga Technical University, Riga, Latvia; \href{{mailto:Nadezda.Kunicina@rtu.lv}}{{Nadezda.Kunicina@rtu.lv}}}}
\corres{{Correspondence: \href{{mailto:islam_md_mahbubul@live.kaznu.kz}}{{islam\_md\_mahbubul@live.kaznu.kz}} (M.M.I.); \href{{mailto:Nadezda.Kunicina@rtu.lv}}{{Nadezda.Kunicina@rtu.lv}} (N.K.)}}

\abstract{{{inline(abstract)}}}
\keyword{{{esc_text(kw)}}}

\begin{{document}}

{body}

\vspace{{6pt}}

{supp_tex}\authorcontributions{{{back['contrib']}}}

\funding{{{back['funding']}}}

\institutionalreview{{{back['irb']}}}

\informedconsent{{{back['consent']}}}

\dataavailability{{{back['data']}}}

\acknowledgments{{{back['ack']}}}

\conflictsofinterest{{{back['coi']}}}

\abbreviations{{Abbreviations}}{{
The following abbreviations are used in this manuscript:\\

\noindent
\begin{{tabular}}{{@{{}}ll}}
{abbr_tex}
\end{{tabular}}
}}

{app_tex}
\begin{{adjustwidth}}{{-\extralength}}{{0cm}}
\reftitle{{References}}
\begin{{thebibliography}}{{999}}
{refs}
\end{{thebibliography}}
\end{{adjustwidth}}
\end{{document}}
"""
    return tex, len(order)


def main():
    tex, n_refs = make_tex()
    (L / "main.tex").write_text(tex, encoding="utf-8")
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], cwd=L,
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode:
            print(r.stdout[-3000:])
            raise SystemExit("pdflatex failed")
    print(n_refs, "references; built", L / "main.pdf")


if __name__ == "__main__":
    main()
