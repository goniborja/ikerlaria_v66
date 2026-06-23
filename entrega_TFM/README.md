# Entrega TFM — Borja Goñi Eguía

Trabajo de Fin de Estudios maquetado sobre la **plantilla oficial de UNIR**
(Máster Universitario en Neuropsicología y Educación – Proyecto de
investigación), a partir del documento de texto exportado a Markdown.

## Archivos

| Archivo | Descripción |
| :-- | :-- |
| `TFM_Borja_Goni_Eguia_UNIR.docx` | **Documento final, listo para entregar.** |
| `TFM_Borja_Goni_Eguia_UNIR_vista_previa.pdf` | Vista previa en PDF (mismo contenido). |
| `pipeline/` | Scripts y fuentes para regenerar el `.docx` si se edita el contenido. |

## Qué se ha respetado de la plantilla

- **Portada** oficial (logo UNIR, titulación, título real, tabla con autor,
  modalidad, directora y fecha).
- **Encabezado** (autor + título abreviado) y **pie de página** con número de página.
- **Numeración automática de apartados** de la plantilla: 1, 1.1, 1.1.1 …
  (el documento original venía con su propia numeración de lista, que se ha
  eliminado para que mande la de la plantilla).
- **Apartados sin numerar** con su estilo propio: Agradecimientos, Resumen,
  Abstract, los tres índices y Referencias bibliográficas.
- **Anexos** con numeración automática «Anexo A.», «Anexo B.».
- **Índice de contenidos, Índice de tablas e Índice de figuras** generados como
  campos automáticos (con números de página reales).
- **Figuras y tablas** con formato APA (rótulo en negrita + título en cursiva +
  *Nota.*), tablas con bordes y la rúbrica a cuerpo reducido para que quepa.
- **Citas**: se han quitado los hipervínculos internos al Google Doc dejando el
  texto de la cita; las referencias llevan **sangría francesa**.
- Las imágenes (figura del córtex y capturas de BOZGORAILUA) van **incrustadas**.

## Actualizar los índices en Word

Los tres índices ya están **rellenados con sus números de página**. Si editas el
documento (añades texto, mueves apartados, etc.) y cambian las páginas, basta con:

1. `Ctrl + A` (seleccionar todo).
2. `F9` (actualizar campos). Si pregunta, elige **«Actualizar toda la tabla»**.

El documento incluye además la opción de *actualizar campos al abrir*, por lo que
Word puede ofrecer hacerlo automáticamente.

## Regenerar el documento (opcional)

Si modificas el contenido, edita el Markdown de `pipeline/source_original.md` y
ejecuta (necesita `pandoc`, `python-docx` y `libreoffice`):

```bash
cd pipeline
bash run_all.sh
```
