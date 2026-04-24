@echo off
REM Ikerlaria v6 Portable - Lanzador Windows
REM Mantiene la ventana abierta si hay error para que el usuario vea el mensaje.
setlocal
cd /d "%~dp0"

echo.
echo ================================================
echo   IKERLARIA v6.1 - DETECTOR IA
echo ================================================
echo.

REM Verificar que Python esta disponible
where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python no esta instalado o no esta en el PATH.
    echo   Instala Python 3.10+ desde https://www.python.org
    echo   IMPORTANTE: durante la instalacion marca "Add Python to PATH".
    echo   Luego ejecuta setup.bat antes de detector.bat.
    echo.
    pause
    exit /b 1
)

REM Verificar que las dependencias basicas estan instaladas
python -c "import flask, spacy, joblib, pandas" 2>nul
if errorlevel 1 (
    echo ERROR: dependencias Python no instaladas.
    echo   Ejecuta setup.bat antes de abrir el detector.
    echo.
    pause
    exit /b 1
)

REM Activar JDK portable si existe.
REM Nota: comillas en `set "PATH=..."` son OBLIGATORIAS — el PATH del sistema
REM suele contener `C:\Program Files (x86)\...` con parentesis que romperian
REM el parseo del bloque `if ( ... )` ("No se esperaba X en este momento").
if exist "%~dp0jdk\bin\java.exe" (
    set "PATH=%~dp0jdk\bin;%PATH%"
    echo   Java portable activado.
)

echo   Arrancando detector en http://localhost:5116
echo   La 1a vez tarda 1-2 min (descarga GPT-2 y LanguageTool).
echo   Cuando veas "Listo. Servidor iniciando" se abrira el navegador solo.
echo   Ctrl+C para parar.
echo.

python src\serve_detector_v6.py %*

echo.
echo ================================================
echo   El servidor se ha detenido.
echo ================================================
pause
