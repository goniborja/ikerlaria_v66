#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assemble the final UNIR TFM .docx from the pandoc-generated body + template."""
import copy, re
import docx
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.enum.text import WD_ALIGN_PARAGRAPH

def w(tag): return qn('w:' + tag)

doc = Document("body.docx")
tpl = Document("plantilla.docx")
body = doc.element.body

REAL_TITLE = ("Conducta de planificación y coherencia textual en escritura "
              "argumentativa en inglés en jóvenes entre 14 y 16 años")
SHORT_TITLE = "Conducta de planificación y coherencia textual en inglés"
AUTHOR = "Borja Goñi Eguía"
DIRECTORA = "Dra. Natalia Louleli"
FECHA = "Arratzu, 18 de marzo de 2026"

FRONT = {"Agradecimientos", "Resumen", "Abstract",
         "Índice de contenidos", "Índice de tablas", "Índice de figuras"}

# ---------------- helpers ----------------
def get_pPr(p):
    pPr = p.find(w('pPr'))
    if pPr is None:
        pPr = OxmlElement('w:pPr'); p.insert(0, pPr)
    return pPr

def style_of(p):
    pPr = p.find(w('pPr'))
    if pPr is None: return "Normal"
    ps = pPr.find(w('pStyle'))
    return ps.get(w('val')) if ps is not None else "Normal"

def set_style(p, sid):
    pPr = get_pPr(p)
    ps = pPr.find(w('pStyle'))
    if ps is None:
        ps = OxmlElement('w:pStyle'); pPr.insert(0, ps)
    ps.set(w('val'), sid)

def text_of(p):
    return "".join(t.text or "" for t in p.iter(w('t')))

def page_break_before(p):
    pPr = get_pPr(p)
    if pPr.find(w('pageBreakBefore')) is None:
        pPr.append(OxmlElement('w:pageBreakBefore'))

def set_jc(p, val):
    pPr = get_pPr(p)
    jc = pPr.find(w('jc'))
    if jc is None:
        jc = OxmlElement('w:jc'); pPr.append(jc)
    jc.set(w('val'), val)

def set_hanging(p):
    pPr = get_pPr(p)
    ind = pPr.find(w('ind'))
    if ind is None:
        ind = OxmlElement('w:ind'); pPr.append(ind)
    ind.set(w('left'), '709'); ind.set(w('hanging'), '709')

def set_runs_text(p, newtext):
    runs = p.findall(w('r'))
    if not runs:
        r = OxmlElement('w:r'); t = OxmlElement('w:t'); t.text = newtext
        r.append(t); p.append(r); return
    first = runs[0]
    for t in first.findall(w('t')):
        first.remove(t)
    t = OxmlElement('w:t'); t.set(qn('xml:space'), 'preserve'); t.text = newtext
    first.append(t)
    for extra in runs[1:]:
        p.remove(extra)

# ---------------- 0) inject missing pandoc style definitions ----------------
styles_root = doc.styles.element
def add_para_style(sid, name, based="Normal", extra_ppr=""):
    xml = f'''<w:style xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" w:type="paragraph" w:customStyle="0" w:styleId="{sid}">
      <w:name w:val="{name}"/><w:basedOn w:val="{based}"/><w:qFormat/>{extra_ppr}</w:style>'''
    styles_root.append(docx.oxml.parse_xml(xml))

# BodyText / FirstParagraph behave like Normal; Compact is tight (for table cells)
add_para_style("BodyText", "Body Text")
add_para_style("FirstParagraph", "First Paragraph")
add_para_style("Compact", "Compact",
    extra_ppr='<w:pPr><w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/><w:jc w:val="left"/></w:pPr>')
# Table style with borders
table_style_xml = '''<w:style xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" w:type="table" w:styleId="Table">
  <w:name w:val="Table"/><w:basedOn w:val="Tablanormal"/>
  <w:tblPr><w:tblBorders>
    <w:top w:val="single" w:sz="4" w:space="0" w:color="000000"/>
    <w:left w:val="single" w:sz="4" w:space="0" w:color="000000"/>
    <w:bottom w:val="single" w:sz="4" w:space="0" w:color="000000"/>
    <w:right w:val="single" w:sz="4" w:space="0" w:color="000000"/>
    <w:insideH w:val="single" w:sz="4" w:space="0" w:color="000000"/>
    <w:insideV w:val="single" w:sz="4" w:space="0" w:color="000000"/>
  </w:tblBorders></w:tblPr></w:style>'''
styles_root.append(docx.oxml.parse_xml(table_style_xml))

# ---------------- 1) style data tables (before cover is inserted) ----------------
for tbl in list(body.iter(w('tbl'))):
    rows = tbl.findall(w('tr'))
    if not rows: continue
    ncols = len(rows[0].findall(w('tc')))
    # header row bold + repeat
    trPr = rows[0].find(w('trPr'))
    if trPr is None:
        trPr = OxmlElement('w:trPr'); rows[0].insert(0, trPr)
    if trPr.find(w('tblHeader')) is None:
        trPr.append(OxmlElement('w:tblHeader'))
    for tc in rows[0].findall(w('tc')):
        for r in tc.findall(w('p')):
            rpr_target = r
        for run in tc.iter(w('r')):
            rPr = run.find(w('rPr'))
            if rPr is None:
                rPr = OxmlElement('w:rPr'); run.insert(0, rPr)
            if rPr.find(w('b')) is None:
                rPr.append(OxmlElement('w:b'))
    # rubric (>=5 cols): shrink font to 9pt for all cells
    if ncols >= 5:
        for run in tbl.iter(w('r')):
            rPr = run.find(w('rPr'))
            if rPr is None:
                rPr = OxmlElement('w:rPr'); run.insert(0, rPr)
            for tag in ('sz', 'szCs'):
                e = rPr.find(w(tag))
                if e is None:
                    e = OxmlElement('w:' + tag); rPr.append(e)
                e.set(w('val'), '18')

# ---------------- 2) walk paragraphs: restyle, breaks, refs, images ----------------
children = list(body.iterchildren())
in_refs = False
past_refs = False
to_remove = []
prev_was_note = False
heading_for_field = {}  # text -> element

for el in children:
    if el.tag != w('p'):
        prev_was_note = False
        continue
    sty = style_of(el)
    txt = text_of(el).strip()

    # headings left-aligned (avoid justified stretching on multi-line headings)
    if sty in ("Ttulo1", "Ttulo2", "Ttulo3", "Anexo", "Ttulondices", "Ttulo1sinnumerar"):
        set_jc(el, "left")

    # remove duplicate consecutive notes
    if sty == "Piedefoto-tabla":
        if prev_was_note:
            to_remove.append(el); continue
        prev_was_note = True
    else:
        prev_was_note = False

    # headings
    if sty == "Ttulo1":
        if txt in FRONT:
            set_style(el, "Ttulondices")
            page_break_before(el)
            if txt in ("Índice de contenidos", "Índice de tablas", "Índice de figuras"):
                heading_for_field[txt] = el
            continue
        if txt.startswith("Referencias"):
            set_style(el, "Ttulo1sinnumerar")
            page_break_before(el)
            in_refs = True
            past_refs = True
            continue
        if past_refs:  # every Ttulo1 after Referencias is an Anexo
            set_style(el, "Anexo")
            page_break_before(el)
            in_refs = False  # the reference list has ended
            continue
        # ordinary numbered chapter
        page_break_before(el)
        continue

    # reference entries: hanging indent, left aligned
    if in_refs and sty in ("BodyText", "FirstParagraph", "Normal") and txt:
        set_hanging(el); set_jc(el, "left")
        continue

    # image paragraphs -> center + scale
    if el.find('.//' + w('drawing')) is not None:
        set_jc(el, "center")

for el in to_remove:
    el.getparent().remove(el)

# ---------------- 3) scale images to fit page width, centered ----------------
MAX_W = int(Cm(15).emu)
for draw in body.iter(w('drawing')):
    # collect cx/cy on wp:extent and a:ext
    exts = []
    for tag in ('{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}extent',
                '{http://schemas.openxmlformats.org/drawingml/2006/main}ext'):
        for e in draw.iter(tag):
            exts.append(e)
    if not exts: continue
    cx = int(exts[0].get('cx'))
    if cx > MAX_W and cx > 0:
        f = MAX_W / cx
        for e in exts:
            e.set('cx', str(int(int(e.get('cx')) * f)))
            e.set('cy', str(int(int(e.get('cy')) * f)))

# ---------------- 4) insert index fields ----------------
def make_field(instr, placeholder):
    p = OxmlElement('w:p')
    def fld(t, dirty=False):
        r = OxmlElement('w:r'); fc = OxmlElement('w:fldChar')
        fc.set(w('fldCharType'), t)
        if dirty: fc.set(w('dirty'), 'true')
        r.append(fc); return r
    r2 = OxmlElement('w:r'); it = OxmlElement('w:instrText')
    it.set(qn('xml:space'), 'preserve'); it.text = instr; r2.append(it)
    r4 = OxmlElement('w:r'); t = OxmlElement('w:t'); t.text = placeholder; r4.append(t)
    for x in (fld('begin', True), r2, fld('separate'), r4, fld('end')):
        p.append(x)
    return p

PLACE = "Actualice este índice: clic derecho ▸ «Actualizar campos» (o seleccione todo y pulse F9)."
fields = {
    "Índice de contenidos": ' TOC \\o "1-3" \\h \\z \\u ',
    "Índice de figuras":    ' TOC \\h \\z \\t "Figuras;1" ',
    "Índice de tablas":     ' TOC \\h \\z \\t "Título de TDC;1" ',
}
for key, instr in fields.items():
    h = heading_for_field.get(key)
    if h is not None:
        h.addnext(make_field(instr, PLACE))

# ---------------- 5) build & prepend cover ----------------
tpl_body = tpl.element.body
cover_els = []
for ch in tpl_body.iterchildren():
    t = "".join(x.text or "" for x in ch.iter(w('t')))
    if ch.tag == w('p') and t.strip() == "Agradecimientos":
        break
    cover_els.append(ch)

# logo rebuilt via add_picture (fresh relationship)
logo_par = doc.add_paragraph()
logo_par.style = doc.styles['No Spacing']
logo_par.alignment = WD_ALIGN_PARAGRAPH.CENTER
logo_par.add_run().add_picture("plantilla_extracted/word/media/image1.png", width=Cm(10.6))
logo_el = logo_par._p
body.remove(logo_el)

# anchor = first front-matter heading (Agradecimientos, now Ttulondices)
anchor = None
for ch in body.iterchildren():
    if ch.tag == w('p') and text_of(ch).strip() == "Agradecimientos":
        anchor = ch; break

# cover_els[0]=leading empty para, [1]=original (now-broken) logo para -> skip both
raw_cover = [logo_el] + [copy.deepcopy(el) for el in cover_els[2:]]
# trim the large empty Normal spacer paragraphs (real title is multi-line) -> keep 2
new_cover = []
empty_normal = 0
for el in raw_cover:
    if el.tag == w('p') and text_of(el).strip() == "" and style_of(el) == "Normal":
        empty_normal += 1
        if empty_normal > 3:
            continue
    new_cover.append(el)
for el in new_cover:
    anchor.addprevious(el)

# fill cover text: title + table cells
for el in new_cover:
    if el.tag == w('p') and text_of(el).strip() == "Título del Trabajo Fin de Estudios":
        set_runs_text(el, REAL_TITLE)
    if el.tag == w('tbl'):
        # make the floating cover table inline so it stays on page 1
        ctblPr = el.find(w('tblPr'))
        if ctblPr is not None:
            tpp = ctblPr.find(w('tblpPr'))
            if tpp is not None:
                ctblPr.remove(tpp)
        for tr in el.findall(w('tr')):
            tcs = tr.findall(w('tc'))
            if len(tcs) < 2: continue
            label = "".join(x.text or "" for x in tcs[0].iter(w('t'))).strip()
            target = None
            if label.startswith("Trabajo fin"): target = AUTHOR
            elif label.startswith("Director"):  target = DIRECTORA
            elif label.startswith("Fecha"):     target = FECHA
            elif label.startswith("Modalidad"): target = "Proyecto de investigación"
            if target is not None:
                # set the paragraph text inside the 2nd cell
                cps = tcs[1].findall(w('p'))
                if cps:
                    set_runs_text(cps[0], target)

# ---------------- 6) header text ----------------
hdr_el = doc.sections[0].header._element
for p in hdr_el.iter(w('p')):
    jt = "".join(t.text or "" for t in p.iter(w('t')))
    if "Nombre y apellidos del estudiante" in jt:
        set_runs_text(p, jt.replace("Nombre y apellidos del estudiante", AUTHOR))
    elif "Título del Trabajo Fin de Estudios" in jt:
        set_runs_text(p, jt.replace("Título del Trabajo Fin de Estudios", SHORT_TITLE))

# ---------------- 7) update fields on open ----------------
settings_el = doc.settings.element
if settings_el.find(w('updateFields')) is None:
    uf = OxmlElement('w:updateFields'); uf.set(w('val'), 'true')
    settings_el.insert(0, uf)

doc.save("TFM_final.docx")
print("Saved TFM_final.docx")
