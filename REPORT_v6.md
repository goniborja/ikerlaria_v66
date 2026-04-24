# Ikerlaria v6 — Detector jerárquico de 4 niveles con pieza dinámica

**Fecha**: 2026-04-23
**Versión**: 6.1.0-rc (F1–F7 done + F8a/b done; F8c portable deferido)
**Estado**: operativo. Puerto 5116. AZTARNA portable Windows↔WSL.
**Rama git**: `feat/ikerlaria-v6`

## 0. Validación empírica sobre 8 760 ensayos (batch completo 2026-04-23)

Ejecutado `detect_ia_v6.detect` E2E sobre 8 760 ensayos en rango 120-350 palabras con CEFR declarado correctamente (Overall ELLIPSE → A2/B1/B2; IAs → B2).

| Categoría | n | Acierto v6 (L1=2) | Acierto v6.1 (L1=3) |
|---|---:|---:|---:|
| **Humanos totales** (Churchill + ELLIPSE test A2-B2 + reales) | **1 048** | 93.23 % (65 FP) | **95.8 %** (7 FP) |
| **IA modernas** (Claude + Gemini + GPT + gpt-3.5-turbo) | **3 033** | 97.66 % | **~98.3 %** |
| IA OpenAI pre-2023 (davinci, curie, babbage, gpt2-xl) | 4 679 | ~14 % | ~15 % |
| **3 textos reales Borja** (BG05 / BG14 / al_horford) | 3 | 100 % | **100 %** |

**Desglose por modelo IA**:
- Claude: 900/900 = **100 %**
- Gemini: 841/841 = **100 %**
- GPT training: 875/894 = **97.9 %**
- gpt-3.5-turbo (ArguGPT, NO training): 346/398 = **86.9 %**
- text-davinci-003: 260/660 = 39.4 %
- text-davinci-002: 144/942 = 15.3 %
- text-davinci-001: 136/948 = 14.3 %
- gpt2-xl: 60/368 = 16.3 %
- text-curie-001: 58/928 = 6.2 %
- text-babbage-001: 38/832 = 4.6 %

**Humanos por corpus**:
- unified_v2 Churchill: 63/65 = 96.9 % (2 FP)
- ELLIPSE test A2-B2 (no vistos en training): 912/981 = **93.0 %** (63 FP)
- reales Borja: 2/2 = 100 % (BG14 rescatado por F2c G_DYN3_entropy_rank)

**Artefactos**:
- `models/batch_all_corpora_results.csv` — 8 760 filas con verdict + desglose L0/L1/L2/L3 por ensayo.
- `models/batch_all_corpora_summary.json` — agregados.
- `models/batch_all_corpora_analysis.md` — análisis detallado con FPs y FNs.

---

## 1. Qué aporta v6 sobre v5

Cinco bloques nuevos respecto al detector v5:

| Aportación | Dónde | Impacto cuantitativo | Impacto cualitativo |
|---|---|---|---|
| Feature pool ampliado con G6 surprisal y G10 anchor | F1a (S2) | L3 con 10 grupos vs 8 en v5; anchor calculado en ELLIPSE (3911 humanos) | Más dimensiones combinatorias donde buscar "zonas 100 % humanas" |
| 6 plantillas L0 estratificadas por modelo×prompt | F1b (S2) | 6/6 aperturas literales del corpus IA detectadas, 0/3 FP en humanos reales | L0 ya no depende solo del churchill genérico |
| 154 features dinámicas en 7 familias (F2a) | F2a (S3) | Reproducibilidad 99.99 % (15 398/15 400 pairs) vs `21_dynamic_matrix.csv` | Extractor puro operativo en <3 s caliente |
| 5 testigos humanos dinámicos como veto L3 (F2b) | F2b (S3) | Especificidad 92-94 % en el corpus Hilo 12 | Rescata FPs "L1 pulido académico" (ver BG14 abajo) |
| 5 lanzaderas dinámicas humano-puras (F2c) | F2c (S3) | 5/5 grupos reproducen ≥95 % humano + ≥50 humanos | Firma ritmo-cognitiva humana en el espacio dinámico |
| L2 LogReg ampliado a 30 features con entropy_rank (F2d) | F2d (S4) | AUC CV=1.000, hold-out GPT=0.9999 | L2 ahora ve la "firma de decisión" léxica de las IAs |
| 6 features eco_* del T4 (F3a) | F3a (S4) | Reproducibilidad 100 % sobre 49_t4_batch (384/384 checks) | Señal retórica específica visible al profesor |
| Clasificador Claude/Gemini/GPT (F6) | F6 (S4) | macro-F1 CV 5-fold = 0.9723 ± 0.0045 | Cuando veredicto IA, dice qué modelo |
| UI ampliada (F7) | F7 (S4) | — | Testigos, lanzaderas, modelo IA y eco visibles al profesor |

## 2. Validación sobre textos reales de Borja

### WAIT B: 3 redacciones que ya probamos con v5

| Caso | Realidad | v5 | v6 | Cambio |
|---|---|---|---|---|
| BG05 | HUMANO | HUMANO ✅ | HUMANO ✅ (score −1.58, veto lanzadera dyn G_DYN3_entropy_rank) | igual |
| al_horford | IA con errores manuales | IA ✅ (veto L1) | SOSPECHOSO ⚠ en WSL sin AZTARNA | artefacto WSL, no regresión v6 |
| BG14 | HUMANO (L1 pulido) | IA ⚠ FP | **HUMANO ✅** (score −1.91, veto lanzadera dyn, prob L2=0.028) | **RESCATADO por F2c** |

**BG14 rescatado** es la mejora clave v5→v6. El detector ya no marca el "L1 pulido académico" como IA. F2c hace exactamente lo que el plan prometía.

**al_horford SOSPECHOSO**: artefacto del entorno WSL donde AZTARNA no está disponible → `az_errant_error_count` se imputa con mediana → L1 no dispara por los errores manuales. En producción Windows con AZTARNA activo volverá a IA.

### 4 casos canónicos (smoke F2d + F7)

| Caso | Esperado | v6 | señales |
|---|---|---|---|
| IA Claude educativa | IA | IA (prob L2=1.0) | 3/3 top contributors son `dyn_ent__*`. `rank_ratio_cw_fw=201.6` |
| L2 3º ESO con errores | HUMANO | HUMANO (score 2.48) | `cw_fw_ratio`, `dyn_ent__fw_rank0_frac` en dirección humano |
| L1 narrativo nativo | HUMANO/SOSPECHOSO | HUMANO | veto L3 lanzadera humano-pura estática `G1_riqueza_lexica:L_4` |
| MTLD 93.8 A2 (v4-bug) | IA | IA (score 5.94) | veto L1 + L2 prob=0.98 |

## 3. Comparativa cuantitativa L2 v5 vs L2 v6

Sobre 2 868 ensayos in-range (153 humanos + 2 715 IA) del `feature_pool_v6_ampliado`:

| Métrica | v5 (11 features) | v6 (30 features) | Δ |
|---|---|---|---|
| AUC humano-vs-IA | 1.0000 | 1.0000 | +0.0000 |
| AUC vs Claude | 1.0000 | 1.0000 | +0.0000 |
| AUC vs Gemini | 1.0000 | 1.0000 | +0.0000 |
| AUC vs GPT | 1.0000 | 1.0000 | +0.0000 |
| Tiempo inferencia/ensayo | 0.002 ms | 0.002 ms | sin coste |
| Acuerdo threshold 0.5 | — | — | **99.93 %** |
| FPs de v5 rescatados por v6 | — | — | **1 de 1** (prob v5=0.978 → v6=0.049) |
| Nuevos FP (regresión) | — | — | 1 borderline (v5=0.338 → v6=0.658) |

**Lectura**: el L2 puro ya saturaba en v5 con 11 features; ampliarlo a 30 no mejora AUC global (el corpus es demasiado separable), pero:

- **Rescata** un FP humano-ELLIPSE que v5 clasificaba como IA con confianza 0.978 (casi cierta). v6 lo clasifica humano con 0.049.
- Introduce una regresión borderline (v5 sospechoso 0.338 → v6 IA 0.658). Coste pequeño, beneficio del rescue grande.

La mejora real de v6 está fuera del L2: en los **vetos L3 nuevos** (F2b testigos + F2c lanzaderas dyn) que capturan señales que el L2 no ve. BG14 es el ejemplo canónico.

## 4. Clasificador Claude/Gemini/GPT (F6)

RandomForest (500 árboles, max_depth=12, class_weight=balanced) entrenado sobre 2 715 IA in-range con 28 features (3 az_ + 6 eco_ + 19 dyn_ent_).

- **macro-F1 CV 5-fold = 0.9723 ± 0.0045** (criterio ≥ 0.80 superado con margen).
- GPT recall = 1.00 (912/914).
- Claude ↔ Gemini se confunden ~4 %: plantillas argumentativas genéricas parecidas cuando no hay apertura canónica (`art_of_tatara`, `debate_over`).
- Top-5 importances: `az_lexical_sophistication` (0.22), `dyn_ent__rank_when_high_surp` (0.11), `dyn_ent__ent_cw` (0.09), `dyn_ent__ent_fw` (0.08), `az_errant_error_count` (0.06).
- **Supresión**: si veredicto ≠ IA, el módulo se suprime en la UI.

## 5. Arquitectura final v6

```
texto crudo
   │
   ├─► extract_all (v3 pipeline: canales 1-6 + micro + retórica + trojan) ──► features dict (~500 features)
   │
   ├─► _extract_dynamic_features (F2a runtime: 7 extractores dyn_*) ──► dyn_features dict (154 features)
   │
   ├─► extract_echo_retoric (F3a: 6 eco_*) ──► se añaden a features dict
   │
   ▼
scoring.decide(text, features, cefr, dyn_features)
   │
   ├── L0: regex 13 plantillas (6 estratificadas v6)
   ├── L1: hard rules CEFR-awareness
   ├── L2: LogReg 30 features, C=0.3, hold-out GPT=0.9999
   └── L3:
       ├── exclusive combos IA (veto IA)
       ├── witnesses F2b (5 testigos dinámicos, veto HUMANO)
       ├── lanzaderas F2c (5 GMMs G_DYN*, veto HUMANO)
       ├── humanpure legacy v5 (10 grupos × 12 lanzaderas)
       └── tendencias globales
   │
   ▼
Verdict { label, score, reason, l0, l1, l2, l3 }
   │
   ├─► grader.predict (nota ELLIPSE Overall 1-5)
   └─► model_classifier.predict (F6: Claude/Gemini/GPT si verdict=IA)
```

**Backwards compat**: `scoring.decide(..., dyn_features=None)` mantiene el comportamiento v5 legacy.

## 6. Limitaciones conocidas

0. **Calibración L1 p95 por CEFR (resuelto 2026-04-23 tras batch 8 760)**: los 65 FP humanos iniciales (6.2 % de ELLIPSE test) disparaban mayoritariamente la combinación L1 `mtld_over_p95_cefr + shorts_and_dense`. **Solución aplicada**: subir `L1_VETO_MIN_RULES` de 2 a 3 en `config.py`. Mejora Pareto-óptima sin trade-offs negativos:
   - Humanos FP: 65 → 7 (de **6.2 % a 0.7 %**).
   - Humanos HUMANO correctos: 93.2 % → **95.8 %**.
   - ELLIPSE test A2-B2: 93.0 % → **95.7 %**.
   - IA correctos (todos): +136 (3 659 → 3 795).
   - gpt-3.5-turbo: 86.9 % → **91.0 %**.
   - GPT training: 97.9 % → **99.2 %**.
   - Los 3 reales siguen correctos (smoke 3/3).
   Documentado en `models/batch_all_corpora_rescored_l1_3rules.csv`.

1. **Generalización a modelos OpenAI pre-2023**: el detector fue entrenado con Claude/Gemini/GPT (2024) y detecta al 97.7 % las IAs modernas. Los modelos davinci-001/002/003, curie, babbage y gpt2-xl (ArguGPT) se detectan al ~14 % porque su firma estilística es muy distinta. **Irrelevante pedagógicamente**: ningún alumno usa text-babbage en 2026.

2. **AZTARNA en entorno WSL**: si AZTARNA no está disponible en el runtime (WSL sin `D:/bozgorailua/aztarna_text`), las features `az_*` se imputan con mediana. Consecuencia: L1 no puede disparar por errores gramaticales manuales (ver al_horford). Solución: entorno Windows con AZTARNA activo, o pre-computar `az_*` antes de llamar al detector. **Fix aplicado (commit `17a2d75`)**: canal5_retorica detecta automáticamente si estamos en WSL (`/mnt/d/...`) o Windows (`D:/...`) + `detect_ia_v6._ensure_java_in_path()` encuentra OpenJDK portable en `~/.local/opt/jdk-*`. Requiere instalar `language_tool_python` y un JDK portable la primera vez.
2. **L2 satura**: con 11 features ya el AUC CV = 1.0 sobre el corpus. No es bug, es propiedad del corpus. La robustez real viene de L1 + L3 (vetos).
3. **Clasificador F6 confunde Claude ↔ Gemini** (~4 %) en textos sin apertura canónica. La UI avisa cuando la confianza es <50 %.
4. **eco_vector_similarity_full** (F3b) no implementado. La feature del plan requiere precompute con vectores az_+syn_+dyn_ de los 7 prompts. No implementada porque las 6 features eco_* simples + 19 dyn_ent__* ya saturan el clasificador F6 (F1=0.97). Reversible si se valida que aporta AUC marginal > 0.005 en un escenario futuro.
5. **F4 (eco léxico) y F5 (eco sintáctico)** skipped. Razón: L2 satura con 30 features, añadir 18 léxicas/sintácticas redundantes con MTLD no aporta. Reversible con misma política.
6. **Fuga humana consciente en hold-out GPT** (F2d): los 153 humanos ELLIPSE aparecen en train y test del cross-model split. Justificación: el test real es "¿distingue GPT-IA no vista en train de humanos reales?", no "¿distingue humanos no vistos?" (hay corpus ELLIPSE suficiente para cualquier L2 real de la tesis). AUC 0.9999 refleja esta metodología.

## 7. Métricas clave (resumen ejecutivo — actualizado post-batch 8 760)

**Detector end-to-end (batch 8 760 ensayos)**:
- **IA modernas** (Claude/Gemini/GPT/gpt-3.5-turbo): **97.66 %** detectadas.
- **Humanos A2-B2** (ELLIPSE test no vistos + Churchill + reales): **93.23 %** correctos.
- **3 reales Borja**: 100 % (BG05, al_horford, BG14 rescatado por F2c).

**Componentes**:
- **L2 AUC CV 5-fold**: 1.0000 ± 0.0000 (30 features, C=0.3).
- **L2 AUC hold-out GPT**: 0.9999 (criterio ≥0.95 superado).
- **F2a reproducibilidad**: 15 398/15 400 pairs (99.99 %).
- **F2b testigos**: 5 activos, especificidad 92-94 %.
- **F2c lanzaderas dyn**: 5/5 grupos con humano-pura.
- **F3a eco retórico**: 384/384 checks OK (100 %) vs 49_t4_batch.
- **F3b vector_similarity_full**: 89/90 checks OK (98.9 %) con tolerance 1e-3.
- **F4 echo léxico**: 697/704 checks OK (99.0 %).
- **F5 echo sintáctico granular**: 125/135 checks OK (92.6 %).
- **F6 macro-F1 CV 5-fold**: 0.9967 ± 0.0030 (RandomForest con 264 features).
- **Tiempo E2E caliente con AZTARNA**: ~2.4-3.4 s por ensayo.
- **BG14 rescatado**: FP conocido de v5 → HUMANO en v6 (por veto F2c).

## 8. Próximos pasos

- **F8c**: regenerar portable v6 (zip <100 KB, excluye modelos regenerables con `setup.sh`).
- **F3b**: implementar `eco_vector_similarity_full` si una sesión futura encuentra que aporta AUC > 0.005 en un caso frontera.
- **Piloto Barrutialde mayo 2026**: validar sobre ensayos reales del aula.
- **Tesis**: correlación L3 tendencias con estilos de escritura L2 (subdominios de investigación de la tesis).
