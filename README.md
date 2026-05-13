# DetecciónCrisis

Sistema de apoyo para líneas de ayuda psicológica que estima en tiempo real el riesgo de crisis emocional en llamadas telefónicas, combinando análisis de texto con un modelo BERT y análisis acústico del audio.

---

## Descripción

DetecciónCrisis procesa archivos de audio de llamadas y produce una puntuación de riesgo (0–1) junto con una etiqueta **alto / medio / bajo**. Está pensado como herramienta de asistencia para supervisores de líneas de crisis: no reemplaza el criterio clínico humano.

El sistema combina dos señales:

| Canal | Técnica | Peso |
|-------|---------|------|
| Texto | BERT multilingual fine-tuned (clasificación suicida/no suicida) | 60 % |
| Audio | Heurística sobre energía RMS y silencios largos | 40 % |

Cuando el riesgo final supera el umbral `high` (por defecto 0.80), se puede enviar una alerta SMS al supervisor vía Twilio.

---

## Arquitectura

```
DetecciónCrisis/
├── config/
│   ├── model_config.json       # Configuración del modelo BERT
│   └── settings.yaml           # Umbrales de riesgo y parámetros de audio
├── models/
│   └── bert-suicide-model/     # Pesos HuggingFace exportados desde Colab
├── notebooks/
│   └── modelobert.py           # Script de entrenamiento (Google Colab)
├── src/
│   ├── main.py                 # CLI
│   ├── modules/
│   │   ├── text_classifier.py  # Clasificación de texto con BERT
│   │   ├── speech_to_text.py   # Transcripción con Whisper
│   │   ├── audio_features.py   # Estimación de riesgo acústico
│   │   └── alerting.py         # Alertas SMS via Twilio
│   ├── services/
│   │   └── realtime_pipeline.py  # Orquesta el flujo completo
│   └── utils/
│       ├── audio_utils.py      # Carga y extracción de características
│       └── risk_rules.py       # Fusión de scores y etiquetado
├── tests/                      # Tests unitarios (pytest)
├── web_app.py                  # Interfaz web Streamlit
├── .env.example                # Plantilla de variables de entorno
└── requirements.txt
```

### Flujo de procesamiento

```
audio.wav
    │
    ├─► Whisper (transcripción ES) ──► BERT ──► text_risk  ─┐
    │                                                         ├─► combine_risk ──► label
    └─► librosa (RMS, silencios)  ──────────────► audio_risk ─┘
                                                              │
                                                         [si "high"] ──► Twilio SMS
```

---

## Requisitos previos

- Python 3.10 o superior
- `ffmpeg` instalado y en el `PATH` (requerido por Whisper)
- Pesos del modelo BERT en `models/bert-suicide-model/` (ver sección *Modelo*)

---

## Instalación

```bash
git clone https://github.com/G-Gamboa/DeteccionCrisis.git
cd DeteccionCrisis

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Configuración

1. Copia `.env.example` a `.env` y rellena las credenciales de Twilio si quieres habilitar alertas:

```bash
cp .env.example .env
```

2. Ajusta los umbrales en `config/settings.yaml` si es necesario:

```yaml
risk_thresholds:
  high: 0.80    # score >= high → "alto riesgo"
  medium: 0.50  # score >= medium → "riesgo medio"
```

---

## Modelo BERT

El modelo no se distribuye en el repositorio por tamaño. Para obtenerlo:

1. Abre `notebooks/modelobert.py` en Google Colab.
2. Ejecuta todas las celdas (requiere cuenta de Kaggle para el dataset).
3. Descarga el ZIP `bert-suicide-model.zip` que genera al final.
4. Descomprime su contenido en `models/bert-suicide-model/`.

La carpeta debe contener al menos: `config.json`, `tokenizer.json`, `tokenizer_config.json`, `vocab.txt` y los pesos del modelo.

---

## Uso

### Interfaz web (Streamlit)

```bash
streamlit run web_app.py
```

Coloca archivos `.wav` en `data/samples/` y selecciónalos desde la barra lateral para analizarlos.

### CLI

```bash
cd src
python main.py --audio /ruta/al/audio.wav
```

Salida de ejemplo:

```
Resultado del análisis (BERT):
  Riesgo texto:    0.823
  Riesgo audio:    0.412
  Riesgo final:    0.659
  Nivel de riesgo: medium
```

---

## Tests

```bash
cd src
pytest ../tests/ -v
```

Los tests en `tests/` cubren la lógica de fusión de riesgo (`risk_rules`), la extracción de características de audio (`audio_utils`) y el estimador acústico (`audio_features`) sin necesidad de cargar el modelo BERT ni Whisper.

---

## Variables de entorno

| Variable | Descripción | Requerida |
|----------|-------------|-----------|
| `ALERTS_ENABLED` | `true` para activar el envío de SMS | No (defecto: `false`) |
| `TWILIO_ACCOUNT_SID` | SID de cuenta Twilio | Solo si alertas activas |
| `TWILIO_AUTH_TOKEN` | Token de autenticación Twilio | Solo si alertas activas |
| `TWILIO_FROM_NUMBER` | Número origen de Twilio | Solo si alertas activas |
| `ALERT_SUPERVISOR_NUMBER` | Número destino del supervisor | Solo si alertas activas |

---

## Limitaciones y consideraciones eticas

- Este sistema es una **herramienta de apoyo**, no un sistema de diagnostico clínico.
- La precisión depende de la calidad del audio y del idioma (optimizado para español).
- El umbral de "alto riesgo" debe calibrarse con datos reales antes de usarlo en producción.
- El modelo BERT fue entrenado con datos en inglés; el rendimiento en español puede variar.
- Todo uso en entornos reales requiere supervisión humana y cumplimiento de la normativa de protección de datos aplicable.

---

## Licencia

MIT
