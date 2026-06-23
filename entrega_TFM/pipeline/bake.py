#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Connect to a headless LibreOffice, open the docx, update all indexes/fields,
save the docx (indexes baked) and export a PDF preview."""
import os, sys, time, uno
from com.sun.star.beans import PropertyValue

def pv(name, val):
    p = PropertyValue(); p.Name = name; p.Value = val; return p

def connect(port, tries=40):
    localContext = uno.getComponentContext()
    resolver = localContext.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", localContext)
    last = None
    for _ in range(tries):
        try:
            ctx = resolver.resolve(
                f"uno:socket,host=localhost,port={port};urp;StarOffice.ComponentContext")
            return ctx
        except Exception as e:
            last = e; time.sleep(0.5)
    raise last

port = 2002
ctx = connect(port)
smgr = ctx.ServiceManager
desktop = smgr.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)

src = os.path.abspath("TFM_final.docx")
url = "file://" + src
doc = desktop.loadComponentFromURL(url, "_blank", 0, (pv("Hidden", True),))

# Update TOC / figure index / table index
idxs = doc.getDocumentIndexes()
for i in range(idxs.getCount()):
    idxs.getByIndex(i).update()
# Update all other fields (page refs, PAGE, etc.)
try:
    doc.getTextFields().refresh()
except Exception:
    pass
doc.refresh()
# update indexes again (page numbers settle after refresh)
for i in range(idxs.getCount()):
    idxs.getByIndex(i).update()

# Save docx with baked indexes
doc.storeToURL(url, (pv("FilterName", "MS Word 2007 XML"),))
# Export PDF preview
doc.storeToURL("file://" + os.path.abspath("TFM_final.pdf"),
               (pv("FilterName", "writer_pdf_Export"),))
n = idxs.getCount()
doc.close(False)
print("baked + pdf OK; indexes:", n)
