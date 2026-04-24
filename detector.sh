#!/bin/bash
# Ikerlaria v6 Portable - Lanzador Linux/Mac/WSL
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

echo ""
echo "================================================"
echo "  IKERLARIA v6.1 - DETECTOR IA"
echo "================================================"
echo ""

# Python disponible?
if ! command -v python3 >/dev/null; then
    echo "ERROR: python3 no esta instalado o no esta en PATH."
    echo "  Instala Python 3.10+ y luego ejecuta ./setup.sh"
    read -p "  Pulsa Enter para cerrar..." _
    exit 1
fi

# Deps basicas?
if ! python3 -c "import flask, spacy, joblib, pandas" 2>/dev/null; then
    echo "ERROR: dependencias Python no instaladas."
    echo "  Ejecuta ./setup.sh antes de detector.sh"
    read -p "  Pulsa Enter para cerrar..." _
    exit 1
fi

# Activar JDK portable si existe
if [ -d "$HERE/jdk/bin" ]; then
    export PATH="$HERE/jdk/bin:$PATH"
    echo "  Java portable activado."
fi

echo "  Arrancando detector en http://localhost:5116"
echo "  La 1a vez tarda 1-2 min (descarga GPT-2 y LanguageTool)."
echo "  Cuando veas 'Listo. Servidor iniciando' se abrira el navegador."
echo "  Ctrl+C para parar."
echo ""

python3 src/serve_detector_v6.py "$@"

echo ""
echo "================================================"
echo "  El servidor se ha detenido."
echo "================================================"
read -p "  Pulsa Enter para cerrar..." _
