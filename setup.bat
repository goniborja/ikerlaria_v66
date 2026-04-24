@echo off
REM Ikerlaria v6 Portable - Setup automatico Windows
REM Instala Python deps + Java portable + modelos spaCy + NLTK

setlocal
cd /d "%~dp0"

echo.
echo ================================================
echo   IKERLARIA v6.1 PORTABLE - SETUP
echo ================================================
echo.
echo Se instalaran (si faltan):
echo   - Dependencias Python (~2 GB incluyendo torch)
echo   - OpenJDK 21 portable (~200 MB en .\jdk\)
echo   - Modelo spaCy en_core_web_md (~55 MB)
echo   - Modelos NLTK: wordnet, stopwords
echo.
echo Durante la primera ejecucion el detector descargara tambien:
echo   - LanguageTool jar (~260 MB)
echo   - GPT-2 small (~500 MB)
echo.
echo Tiempo total estimado: 15-30 min.
echo.
pause

REM ---------- Python ----------
echo.
echo [1/5] Verificando Python...
where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python no encontrado. Instala desde https://www.python.org
    pause
    exit /b 1
)
python --version

REM ---------- pip deps ----------
echo.
echo [2/5] Instalando paquetes Python...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

REM ---------- Java portable ----------
echo.
echo [3/5] Verificando Java...
where java >nul 2>&1
if %errorlevel% equ 0 (
    echo Java ya instalado.
    java -version
) else (
    if exist "%~dp0jdk\bin\java.exe" (
        echo JDK portable ya presente en .\jdk\
    ) else (
        echo Descargando OpenJDK 21 portable para Windows x64...
        powershell -Command "Invoke-WebRequest -Uri 'https://download.java.net/java/GA/jdk21.0.2/f2283984656d49d69e91c558476027ac/13/GPL/openjdk-21.0.2_windows-x64_bin.zip' -OutFile 'jdk.zip'"
        powershell -Command "Expand-Archive -Path 'jdk.zip' -DestinationPath '.'"
        del jdk.zip
        REM Renombrar jdk-21.0.2 a jdk
        if exist jdk-21.0.2 ren jdk-21.0.2 jdk
        echo JDK extraido en .\jdk\
    )
)

REM ---------- spaCy model ----------
echo.
echo [4/5] Descargando modelo spaCy en_core_web_md (~55 MB)...
python -m spacy download en_core_web_md

REM ---------- NLTK resources ----------
echo.
echo [5/5] Descargando recursos NLTK...
python -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('omw-1.4', quiet=True); print('OK')"

echo.
echo ================================================
echo   SETUP COMPLETO
echo ================================================
echo.
echo Arrancar el detector:
echo   detector.bat
echo.
echo La primera ejecucion tardara ~1-2 min porque descargara
echo GPT-2 (~500 MB) y LanguageTool jar (~260 MB) la primera vez.
echo.
pause
