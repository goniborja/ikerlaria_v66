#!/bin/bash
# Reproducible pipeline: source_original.md (+ plantilla.docx) -> TFM_final.docx + PDF
# Requires: pandoc, python3 + python-docx, libreoffice (con módulo Writer y python-uno).
set -e
cd "$(dirname "$0")"
ABS=$PWD

# 1) Limpiar el Markdown exportado
python3 preprocess.py

# 2) Convertir el cuerpo con pandoc usando la plantilla como referencia de estilos
pandoc clean.md -o body.docx --reference-doc=plantilla.docx -f markdown-implicit_figures

# 3) Descomprimir la plantilla (necesario para recuperar el logo de la portada)
rm -rf plantilla_extracted && unzip -oq plantilla.docx -d plantilla_extracted

# 4) Ensamblar: portada, estilos, índices, anexos, encabezado, etc.
python3 assemble.py

# 4b) Fijar márgenes a 2 cm (requisito de formato del TFE)
python3 set_margins.py

# 5) Rellenar los índices (TOC / figuras / tablas) y exportar PDF con LibreOffice
pkill -9 soffice 2>/dev/null || true
sleep 1
soffice --headless --norestore --invisible --nodefault --nologo \
  "--accept=socket,host=localhost,port=2002;urp;" \
  -env:UserInstallation=file://$ABS/louno >/dev/null 2>&1 &
sleep 3
python3 bake.py
pkill -9 soffice 2>/dev/null || true

echo "Listo -> TFM_final.docx (+ TFM_final.pdf)"
