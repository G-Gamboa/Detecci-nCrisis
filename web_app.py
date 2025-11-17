# web_app.py
import sys
from pathlib import Path

import streamlit as st

# Asegurar que podamos importar desde src/
BASE_DIR = Path(__file__).parent
SRC_DIR = BASE_DIR / "src"
sys.path.append(str(SRC_DIR))

from services.realtime_pipeline import RealtimePipeline  # noqa


# Inicializar pipeline una sola vez
@st.cache_resource
def get_pipeline():
    return RealtimePipeline()


def get_sample_files():
    samples_root = BASE_DIR / "data" / "samples"
    if not samples_root.exists():
        return []
    return sorted(samples_root.rglob("*.wav"))


def risk_color(label: str) -> str:
    if label == "high":
        return "#b00020"  # rojo oscuro
    if label == "medium":
        return "#ff8f00"  # ámbar
    return "#2e7d32"      # verde


def main():
    st.set_page_config(
        page_title="Detector de Crisis Emocionales",
        layout="centered"
    )

    st.title("Detector de Crisis Emocionales")
    st.caption("Versión BERT + análisis de audio (demo visual)")

    pipeline = get_pipeline()
    wav_files = get_sample_files()

    if not wav_files:
        st.error(
            "No se encontraron archivos .wav en `data/samples`. "
            "Colocá tus audios allí y recargá la página."
        )
        return

    # Mostrar lista de audios relativa a data/samples
    samples_root = BASE_DIR / "data" / "samples"
    options = []
    for f in wav_files:
        rel = f.relative_to(samples_root)
        options.append((str(rel), f))

    st.sidebar.header("Configuración")
    selected_label = st.sidebar.selectbox(
        "Seleccioná un audio de prueba",
        [label for label, _ in options],
        index=0
    )

    selected_path = dict(options)[selected_label]

    st.subheader("Audio seleccionado")
    st.write(f"`{selected_label}`")

    # Reproductor de audio
    with open(selected_path, "rb") as f:
        audio_bytes = f.read()
    st.audio(audio_bytes, format="audio/wav")

    if st.button("Analizar audio"):
        with st.spinner("Procesando audio con el pipeline..."):
            result = pipeline.process_file(str(selected_path))

        st.subheader("Resultados del análisis")

        # Riesgos numéricos
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Riesgo texto", f"{result.text_risk:.3f}")
        with col2:
            st.metric("Riesgo audio", f"{result.audio_risk:.3f}")
        with col3:
            st.metric("Riesgo final", f"{result.final_risk:.3f}")

        # Barra de riesgo final
        st.markdown("### Nivel de riesgo global")
        color = risk_color(result.risk_label)
        pct = int(result.final_risk * 100)

        st.markdown(
            f"""
            <div style="
                width: 100%;
                border-radius: 8px;
                border: 1px solid #ccc;
                padding: 8px 12px;
            ">
                <div style="font-weight: 600; margin-bottom: 4px;">
                    Nivel: <span style="color: {color}; text-transform: uppercase;">
                        {result.risk_label}
                    </span>
                </div>
                <div style="
                    background-color: #eee;
                    border-radius: 4px;
                    overflow: hidden;
                    height: 18px;
                ">
                    <div style="
                        width: {pct}%;
                        height: 100%;
                        background-color: {color};
                    "></div>
                </div>
                <div style="margin-top: 4px; font-size: 0.9rem;">
                    Puntaje: {result.final_risk:.3f} (0 = sin riesgo, 1 = riesgo máximo)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Texto (transcripción simulada)
        st.markdown("### Transcripción estimada")
        st.write(result.text)

        # Features de audio
        if result.extra and "audio_features" in result.extra:
            feats = result.extra["audio_features"]
            st.markdown("### Rasgos de audio extraídos")
            st.write(
                {
                    "rms": feats.get("rms"),
                    "zcr": feats.get("zcr"),
                    "long_silences_ratio": feats.get("long_silences_ratio"),
                }
            )

if __name__ == "__main__":
    main()
