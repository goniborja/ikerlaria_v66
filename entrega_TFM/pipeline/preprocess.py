#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preprocess the Pandoc-exported TFM markdown into a clean markdown that maps
cleanly onto the UNIR template via pandoc --reference-doc."""
import re, sys

SRC = "source_original.md"
OUT = "clean.md"

raw = open(SRC, encoding="utf-8").read()
lines = raw.split("\n")

# 1) Separate the trailing image reference definitions ([imageN]: <data:...>)
img_defs = []
body_lines = []
for ln in lines:
    if re.match(r'^\[image\d+\]:\s', ln):
        img_defs.append(ln)
    else:
        body_lines.append(ln)

# 2) Strip ordered-list prefixes from headings:  "7. # X" / "   1. ## Y" -> "# X" / "## Y"
def strip_heading_num(ln):
    m = re.match(r'^\s*\d+\.\s+(#{1,6})\s+(.*)$', ln)
    if m:
        ln = m.group(1) + " " + m.group(2).strip()
    # strip surrounding bold markers from any heading text
    m2 = re.match(r'^(#{1,6})\s+\*\*(.*?)\*\*\s*$', ln)
    if m2:
        ln = m2.group(1) + " " + m2.group(2).strip()
    return ln
body_lines = [strip_heading_num(ln) for ln in body_lines]

# 3) Drop everything before the first heading "# Agradecimientos" (the cover block;
#    we rebuild the cover from the template).
start = 0
for i, ln in enumerate(body_lines):
    if re.match(r'^#\s+Agradecimientos', ln):
        start = i
        break
body_lines = body_lines[start:]

text = "\n".join(body_lines)

# 4) Remove Google-Docs citation hyperlinks, keep the visible text:  [txt](http...docs.google...) -> txt
text = re.sub(r'\[([^\]]+)\]\((?:https?:)?//[^)]*google[^)]*\)', r'\1', text)
text = re.sub(r'\[([^\]]+)\]\((?:https?:)?//docs\.google[^)]*\)', r'\1', text)
# any remaining markdown links -> keep text (none expected except above)
text = re.sub(r'\[([^\]]+)\]\((?:https?:)?//[^)\s]+\)', r'\1', text)

# 5) Normalise image alt text to empty so pandoc inserts a plain inline image
text = re.sub(r'!\[[^\]]*\]\[(image\d+)\]', r'![](\1)', text)
# convert reference-style image defs to inline usage: replace ![](imageN) with the data uri later.
# Build a dict of image -> data uri
imgmap = {}
for d in img_defs:
    m = re.match(r'^\[(image\d+)\]:\s*<?([^>]+)>?\s*$', d)
    if m:
        imgmap[m.group(1)] = m.group(2).strip()
# Replace ![](imageN) with ![](<datauri>)
def repl_img(m):
    key = m.group(1)
    uri = imgmap.get(key, "")
    return f'![]({uri})'
text = re.sub(r'!\[\]\((image\d+)\)', repl_img, text)

lines = text.split("\n")

# 6) Walk the lines, handling front-matter index sections and figure/table captions.
out = []
i = 0
n = len(lines)

FRONT_INDEX = {"Índice de contenidos", "Índice de tablas", "Índice de figuras"}

def is_heading(ln):
    return re.match(r'^#{1,6}\s+', ln)

def heading_text(ln):
    m = re.match(r'^#{1,6}\s+(.*)$', ln)
    return m.group(1).strip() if m else ""

while i < n:
    ln = lines[i]

    # --- Anexo headings: strip "Anexo X." prefix (style auto-numbers) ---
    hm = re.match(r'^(#\s+)Anexo\s+[A-Z]\.\s*(.*)$', ln)
    if hm:
        out.append(hm.group(1) + hm.group(2).strip())
        i += 1
        continue

    # --- Front-matter index sections: keep heading, drop the manual list under them ---
    if is_heading(ln) and heading_text(ln) in FRONT_INDEX:
        out.append(ln)
        i += 1
        # skip until next heading
        while i < n and not is_heading(lines[i]):
            i += 1
        out.append("")  # spacing
        continue

    # --- Figure caption block: a line that is exactly "Figura N" / "Figura BN" ---
    fm = re.match(r'^(Figura)\s+(B?\d+)\s*$', ln.strip())
    tm = re.match(r'^(Tabla)\s+(\d+)\s*$', ln.strip())
    if fm or tm:
        label = (fm or tm).group(1)
        num = (fm or tm).group(2)
        is_anexo_fig = bool(fm and num.startswith("B"))
        # find the title = next non-empty line
        j = i + 1
        while j < n and lines[j].strip() == "":
            j += 1
        title = lines[j].strip() if j < n else ""
        # strip surrounding * or ** from the title
        title = re.sub(r'^\*+\s*', '', title)
        title = re.sub(r'\s*\*+$', '', title).strip()
        # choose style
        if label == "Figura" and not is_anexo_fig:
            style = "Figuras"
        elif label == "Tabla":
            style = "Título de TDC"
        else:
            style = None  # anexo figures: plain bold+italic, not indexed
        cap = f"**{label} {num}**. *{title}*"
        if style:
            out.append("")
            out.append(f'::: {{custom-style="{style}"}}')
            out.append(cap)
            out.append(":::")
            out.append("")
        else:
            out.append("")
            out.append(cap)
            out.append("")
        i = j + 1
        continue

    # --- Nota lines under figures/tables -> Pie de foto-tabla ---
    nm = re.match(r'^\*?Nota\.', ln.strip())
    if nm:
        nota = ln.strip()
        # normalise: ensure "Nota." is italic per APA -> *Nota.* rest
        nota = re.sub(r'^\*?Nota\.\*?\s*', '', nota)
        nota = nota.strip().rstrip('*').strip()
        out.append("")
        out.append(':::: {custom-style="Pie de foto-tabla"}')
        out.append(f'*Nota.* {nota}')
        out.append("::::")
        out.append("")
        i += 1
        continue

    out.append(ln)
    i += 1

text = "\n".join(out)

# 7) Remove duplicate stray lines after the rubric table in Anexo A
#    (the standalone repeated title and a second nota).
text = text.replace(
    "Rúbrica analítica de coherencia y cohesión textual para redacciones argumentativas en inglés L2 (nivel A2-B1)\n",
    "")

# 8) Collapse 3+ blank lines
text = re.sub(r'\n{3,}', '\n\n', text)

open(OUT, "w", encoding="utf-8").write(text)
print("Wrote", OUT, "chars:", len(text))
# quick heading listing
for ln in text.split("\n"):
    if re.match(r'^#{1,6}\s+', ln):
        print("  H:", ln[:80])
