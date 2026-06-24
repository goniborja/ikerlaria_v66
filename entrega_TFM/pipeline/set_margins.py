#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fija los márgenes del documento ensamblado a 2 cm (1134 twips) en todos los
lados, conforme a los 'Requisitos generales de formato' de las instrucciones del
TFE ('márgenes de 2 cm en todo el texto, constantes'). Se aplica sobre
TFM_final.docx para no alterar la plantilla original (plantilla.docx)."""
import zipfile, re, shutil, os

DOCX = "TFM_final.docx"
TMP = "_margins_tmp"
TWIPS_2CM = 1134  # 2 cm = 1134 twips

shutil.rmtree(TMP, ignore_errors=True)
with zipfile.ZipFile(DOCX) as z:
    z.extractall(TMP)

p = f"{TMP}/word/document.xml"
d = open(p, encoding="utf-8").read()

def fix(m):
    s = m.group(0)
    for k in ("top", "bottom", "left", "right"):
        s = re.sub(r'w:%s="-?\d+"' % k, 'w:%s="%d"' % (k, TWIPS_2CM), s)
    return s

d, n = re.subn(r'<w:pgMar[^/]*/>', fix, d)
open(p, "w", encoding="utf-8").write(d)

os.remove(DOCX)
with zipfile.ZipFile(DOCX, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(TMP):
        for fn in files:
            fp = os.path.join(root, fn)
            z.write(fp, os.path.relpath(fp, TMP))
shutil.rmtree(TMP, ignore_errors=True)
print(f"margins set to 2 cm on {n} section(s)")
