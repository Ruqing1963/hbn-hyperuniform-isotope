#!/usr/bin/env python3
r"""
_mkpreview.py -- build a readable PDF WITHOUT REVTeX 4.2.

Two REVTeX behaviours have no article-class equivalent and are rewritten here
rather than worked around in TeX:

  1. \author accumulates in REVTeX; in article the last one wins and every
     earlier co-author is silently dropped.  All author/email/affiliation
     groups are folded into a single \author block with numbered affiliations.
  2. REVTeX puts \begin{abstract} before \maketitle; article typesets it
     there, i.e. above the title.  The abstract block is moved after
     \maketitle.

The output is NOT the REVTeX typesetting -- line breaking, float placement and
the bibliography all differ.  Use `make paper` with texlive-publishers for the
submission version.
"""
import re, pathlib

s = pathlib.Path('paper.tex').read_text()
shim = pathlib.Path('revtex_shim.tex').read_text()
bib = pathlib.Path('refs.bib').read_text()

# ---- 1. fold authors -------------------------------------------------------
pat = re.compile(r'\\author\{(.*?)\}\s*(?:\\email\{(.*?)\}\s*)?\\affiliation\{(.*?)\}',
                 re.S)
groups = pat.findall(s)
affs, names = [], []
for name, mail, aff in groups:
    aff = ' '.join(aff.split())
    if aff not in affs:
        affs.append(aff)
    tag = affs.index(aff) + 1
    mail = f"\\,\\textsuperscript{{*}}" if False else ""
    names.append(f"{' '.join(name.split())}\\textsuperscript{{{tag}}}")
block = ", ".join(names) + r" \\[4pt] " + r" \\ ".join(
    f"\\textsuperscript{{{i+1}}}\\textit{{{a}}}" for i, a in enumerate(affs))
mails = ", ".join(m.strip() for _, m, _ in groups if m.strip())
if mails:
    block += r" \\[2pt] \small " + mails
s = pat.sub("", s, count=len(groups))
s = s.replace(r'\date{\today}', "\\author{" + block + "}\n\\date{\\today}")

# ---- 2. move the abstract below \maketitle ---------------------------------
m = re.search(r'\\begin\{abstract\}.*?\\end\{abstract\}', s, re.S)
if m:
    s = s.replace(m.group(0), "")
    s = s.replace(r'\maketitle', r'\maketitle' + "\n\n" + m.group(0), 1)

# ---- 3. inline the bibliography, ordered by first citation ------------------
meta = {}
for blk in re.split(r'\n(?=@)', bib):
    h = re.match(r'@(\w+)\{([^,]+),', blk)
    if not h:
        continue
    f = {fm.group(1): ' '.join(fm.group(2).split())
         for fm in re.finditer(r'(\w+)\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}',
                               blk[h.end():], re.S)}
    meta[h.group(2).strip()] = (h.group(1), f)

order = []
for m2 in re.finditer(r'\\cite\{(.+?)\}', s):
    for k in (x.strip() for x in m2.group(1).split(',')):
        if k not in order:
            order.append(k)
missing = [k for k in order if k not in meta]
if missing:
    raise SystemExit(f"citations with no bib entry: {missing}")

clean = lambda x: x.replace('{', '').replace('}', '')

def fmt(k):
    typ, f = meta[k]
    au = clean(f.get('author', '')).replace(' and ', ', ')
    t = clean(f.get('title', ''))
    if typ == 'book':
        return (f"{au}, \\textit{{{t}}} ({clean(f.get('publisher',''))}, "
                f"{clean(f.get('address',''))}, {f.get('year','')}).")
    if typ == 'incollection':
        return (f"{au}, {t}, in \\textit{{{clean(f.get('booktitle',''))}}}, "
                f"{clean(f.get('series',''))} \\textbf{{{f.get('volume','')}}} "
                f"({clean(f.get('publisher',''))}, {f.get('year','')}), "
                f"pp. {f.get('pages','')}.")
    if typ == 'misc':
        return f"{au}, {t}, Zenodo ({f.get('year','')}), doi: {f.get('doi','')}."
    v = f" \\textbf{{{f['volume']}}}," if 'volume' in f else ''
    return (f"{au}, {t}, \\textit{{{clean(f.get('journal',''))}}}{v} "
            f"{f.get('pages','')} ({f.get('year','')}).")

items = "\n".join(f"\\bibitem{{{k}}} {fmt(k)}" for k in order)
body = s[s.index(r'\begin{document}'):].replace(
    r'\bibliography{refs}',
    "\\begin{thebibliography}{99}\n" + items + "\n\\end{thebibliography}")

pathlib.Path('preview.tex').write_text(
    shim + "\n"
    r"\newcommand{\tenB}{$^{10}$B}" "\n"
    r"\newcommand{\elevenB}{$^{11}$B}" "\n"
    r"\newcommand{\kzero}{k_{0}}" "\n"
    r"\newcommand{\Tr}{\mathcal{T}}" "\n" + body)
print(f"authors folded: {len(groups)} -> 1 block, {len(affs)} affiliations")
print(f"abstract moved below \\maketitle: {bool(m)}")
print(f"{len(order)} references inlined")
