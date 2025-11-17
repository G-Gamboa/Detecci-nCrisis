import argparse
import pathlib
from dotenv import load_dotenv

from services.realtime_pipeline import RealtimePipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sistema de Detección de Crisis Emocionales (Versión BERT)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single-audio",
        choices=["single-audio"],
        help="Modo de ejecución del sistema."
    )
    parser.add_argument(
        "--audio",
        type=str,
        help="Ruta a archivo de audio para modo single-audio."
    )
    return parser.parse_args()


def run_single_audio(pipeline: RealtimePipeline, audio_path: str):
    audio_file = pathlib.Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"No se encontró el archivo de audio: {audio_file}")
    result = pipeline.process_file(str(audio_file))
    print("Resultado del análisis (BERT):")
    print(f"  Riesgo texto:    {result.text_risk:.3f}")
    print(f"  Riesgo audio:    {result.audio_risk:.3f}")
    print(f"  Riesgo final:    {result.final_risk:.3f}")
    print(f"  Nivel de riesgo: {result.risk_label}")


def main():
    load_dotenv()
    args = parse_args()
    pipeline = RealtimePipeline()
    if args.mode == "single-audio":
        if not args.audio:
            raise SystemExit("Debe proporcionar --audio en modo single-audio.")
        run_single_audio(pipeline, args.audio)


if __name__ == "__main__":
    main()
