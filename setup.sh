#!/bin/bash
# Ikerlaria v6 Portable - Setup automatico Linux/Mac/WSL
# Instala Python deps + Java portable + modelos spaCy + NLTK
set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

echo ""
echo "================================================"
echo "  IKERLARIA v6.1 PORTABLE - SETUP"
echo "================================================"
echo ""
echo "Se instalaran (si faltan):"
echo "  - Dependencias Python (~2 GB incluyendo torch)"
echo "  - OpenJDK 21 portable (~200 MB, en ./jdk/)"
echo "  - Modelo spaCy en_core_web_md (~55 MB)"
echo "  - Modelos NLTK: wordnet, stopwords"
echo ""
echo "Durante la primera ejecucion el detector descargara tambien:"
echo "  - LanguageTool jar (~260 MB)"
echo "  - GPT-2 small (~500 MB)"
echo ""
echo "Tiempo total estimado: 15-30 min (depende de tu conexion)."
echo ""
read -p "Pulsa Enter para continuar..." _

# ---------- Python ----------
echo ""
echo "[1/5] Verificando Python..."
if ! command -v python3 >/dev/null; then
    echo "ERROR: Python3 no encontrado. Instalalo desde https://www.python.org"
    exit 1
fi
python3 --version
echo ""

# ---------- pip deps ----------
echo "[2/5] Instalando paquetes Python..."
python3 -m pip install --upgrade pip
python3 -m pip install --break-system-packages --user -r requirements.txt
echo ""

# ---------- Java portable ----------
echo "[3/5] Verificando Java..."
if command -v java >/dev/null; then
    echo "Java ya instalado:"
    java -version 2>&1 | head -1
else
    JDK_DIR="$HERE/jdk"
    if [ -d "$JDK_DIR/bin" ]; then
        echo "JDK portable ya presente en $JDK_DIR"
    else
        echo "Descargando OpenJDK 21 portable (~200 MB)..."
        OS=$(uname -s | tr '[:upper:]' '[:lower:]')
        ARCH=$(uname -m)
        case "$OS-$ARCH" in
            linux-x86_64)
                URL="https://download.java.net/java/GA/jdk21.0.2/f2283984656d49d69e91c558476027ac/13/GPL/openjdk-21.0.2_linux-x64_bin.tar.gz"
                ;;
            darwin-x86_64|darwin-arm64)
                URL="https://download.java.net/java/GA/jdk21.0.2/f2283984656d49d69e91c558476027ac/13/GPL/openjdk-21.0.2_macos-${ARCH//arm64/aarch64}_bin.tar.gz"
                ;;
            *)
                echo "WARNING: OS/arch no soportado para auto-download. Instala Java 21 manualmente."
                exit 1
                ;;
        esac
        cd "$HERE"
        wget -q "$URL" -O jdk.tar.gz
        tar xzf jdk.tar.gz
        rm jdk.tar.gz
        # Renombrar a jdk/ (la carpeta extraida suele ser jdk-21.0.2)
        mv jdk-21* jdk
        echo "JDK extraido en $JDK_DIR"
    fi
fi
echo ""

# ---------- spaCy model ----------
echo "[4/5] Descargando modelo spaCy en_core_web_md (~55 MB)..."
python3 -m spacy download en_core_web_md
echo ""

# ---------- NLTK resources ----------
echo "[5/5] Descargando recursos NLTK (wordnet + stopwords)..."
python3 -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('omw-1.4', quiet=True); print('OK')"
echo ""

echo "================================================"
echo "  SETUP COMPLETO"
echo "================================================"
echo ""
echo "Arrancar el detector:"
echo "  ./detector.sh"
echo ""
echo "La primera ejecucion tardara ~1-2 min porque descargara"
echo "GPT-2 (~500 MB) y LanguageTool jar (~260 MB) la primera vez."
echo ""
