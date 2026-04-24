# Ikerlaria v6.1 — Detector de IA en redacciones L2 (Portable)

Herramienta para profesores de inglés L2 que detecta si una redacción ha sido escrita por un alumno o generada por una IA (Claude, Gemini, GPT). Diseñada como parte de la tesis doctoral de Borja Goñi (UPV/EHU).

## Métricas empíricas

Sobre 8 760 ensayos del corpus completo (validación 2026-04-23):

| Qué | Acierto |
|---|---|
| **IAs modernas** (Claude, Gemini, GPT, gpt-3.5-turbo) | **~98 %** |
| **Humanos A2-B2** (ELLIPSE no vistos en training) | **95.8 %** |
| 3 textos reales Borja (BG05, BG14, al_horford) | 100 % |
| IAs OpenAI pre-2023 (davinci-001/002, curie, babbage, gpt2-xl) | ~15 % (limitación conocida — firma estilística muy antigua) |

## Instalación (1ª vez, ~15-30 min)

### Linux / Mac / WSL
```bash
./setup.sh
```

### Windows
```
setup.bat
```

El setup instala:
- Dependencias Python (pip).
- OpenJDK 21 portable en `./jdk/` (necesario para LanguageTool).
- Modelo spaCy `en_core_web_md` (~55 MB).
- Recursos NLTK (wordnet, stopwords).

La **primera ejecución del detector** descarga además:
- LanguageTool jar (~260 MB).
- GPT-2 small (~500 MB, para surprisal).

Quedan cacheados en el home del usuario, **no se descargan cada vez**.

## Uso

```bash
./detector.sh     # Linux/Mac/WSL
detector.bat      # Windows
```

Abre un servidor local Flask en **http://localhost:5116**. La interfaz web permite:
1. Pegar la redacción del alumno.
2. Declarar el nivel CEFR (A2/B1/B2/C1).
3. Pulsar "Analizar" — veredicto en ~1-3 s.

El profesor ve:
- **Verdict** (IA / SOSPECHOSO / HUMANO) con explicación en castellano llano.
- **Nota estimada** (escala ELLIPSE 1.0-5.0 + equivalente CEFR) si es humano.
- **Modelo de IA identificado** (Claude / Gemini / GPT) con confianza si es IA.
- **4 análisis paralelos** (L0/L1/L2/L3) con las señales concretas que disparan.

## Arquitectura

El detector combina 4 niveles jerárquicos con vetos:

- **L0** — Detecta plantillas canónicas de aperturas (13 patrones regex).
- **L1** — Reglas hard CEFR-aware (MTLD, errores, léxico). ≥3 reglas → veto IA.
- **L2** — LogReg con 30 features (estáticas + dinámicas entropy_rank).
- **L3** — Combinatoria de lanzaderas GMM sobre 15 grupos de features. Incluye:
  - 41 combos IA-exclusivas → veto IA.
  - 12 lanzaderas humano-puras estáticas + 5 dinámicas (F2c) → veto HUMANO.
  - 5 testigos dinámicos (F2b, especificidad 92-94 %) → veto HUMANO si ≥2 disparan.

Plus:
- **F3 eco retórico** (7 features sobre prompt/output similarity).
- **F4 eco léxico** (15 features literal/parafrástico/BPE).
- **F5 eco sintáctico** (3 features trigramas funcionales).
- **F6 clasificador multiclase** Claude/Gemini/GPT (macro-F1 = 0.997).

Detalles completos en `REPORT_v6.md`.

## Troubleshooting

### "Error: Failed to fetch" o alert al pulsar "Analizar" en el navegador
El servidor Flask no está corriendo o no respondió. Abre la ventana de terminal donde ejecutaste `detector.sh`/`detector.bat`: ahí aparecerá el mensaje de error concreto. Causas más frecuentes:

- **"No java install detected"** → falta el JDK portable. Re-ejecuta `setup.sh`/`setup.bat`.
- **"en_core_web_md no encontrado"** → ejecuta `python -m spacy download en_core_web_md`.
- **"Address already in use: 5116"** → cierra otra ventana del detector o reinicia el PC.
- **Ventana de cmd se cerró sola** → algún error de importación. Abre `cmd` manualmente, navega al portable y ejecuta `python src\serve_detector_v6.py` para ver el traceback.

### "ModuleNotFoundError" al arrancar
Dependencias pip no instaladas. Re-ejecuta `setup.bat`/`setup.sh`.

### El navegador no se abre solo
Abre manualmente: **http://localhost:5116**

### La 1ª ejecución parece colgada (2-3 min sin output)
Normal. Está descargando GPT-2 (~500 MB) y LanguageTool jar (~260 MB). Quedan cacheados; las siguientes ejecuciones arrancan en ~30 s.

### Python no está en PATH (Windows)
Reinstala Python 3.10+ **marcando "Add Python to PATH"** en el instalador. Reinicia cmd.

## Limitaciones conocidas

1. **Idioma**: solo inglés. Para euskera/castellano hay que entrenar desde cero.
2. **Rango**: 120-500 palabras recomendado. Fuera de rango, la confianza baja.
3. **Modelos OpenAI pre-2023**: el detector fue entrenado con Claude/Gemini/GPT modernos. Los modelos davinci/curie/babbage son irrelevantes pedagógicamente pero documentados.
4. **Primera ejecución lenta**: ~1-2 min mientras descarga GPT-2 y LanguageTool.

## Estructura del portable

```
ikerlaria_v6_portable/
├── README.md                  <- este archivo
├── REPORT_v6.md               <- informe técnico completo
├── requirements.txt
├── setup.sh / setup.bat       <- instalador (1ª vez)
├── detector.sh / detector.bat <- lanzador
├── detector.html              <- UI web
├── src/                       <- código del detector
├── aztarna_text/              <- motor retórico
├── ikerlaria_v3/extractors/   <- pipeline base
├── models/                    <- modelos pre-entrenados (~13 MB)
│   ├── logreg_l2_v6.pkl
│   ├── model_classifier_v6_full.pkl
│   ├── gmm_lanzaderas_v6/*.pkl
│   ├── prompts_vectorized.pkl
│   └── ...
├── tests/                     <- smoke tests
└── jdk/                       <- OpenJDK 21 (creado por setup)
```

## Créditos

- **Autor**: Borja Goñi (goniborja@gmail.com)
- **Proyecto**: Bozgorailua / tesis doctoral UPV/EHU
- **Implementación asistida**: Claude Code
- **Corpus**: ELLIPSE + Berdintasuna + ArguGPT

## Licencia

Uso educativo y de investigación. No redistribuir sin permiso.
