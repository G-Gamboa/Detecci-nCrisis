import whisper
from typing import Optional

class SpeechToTextEngine:
    def __init__(self, model_name: str = "small"):
        """
        Modelos recomendados:
        tiny  -> muy rápido, menos precisión
        base  -> rápido y decente
        small -> ideal para español
        medium -> muy bueno
        large-v3 -> el mejor (pero pesado)
        """
        self.model_name = model_name
        self.model = None

    def load(self):
        if self.model is None:
            print(f"[ASR] Cargando modelo Whisper: {self.model_name} ...")
            self.model = whisper.load_model(self.model_name)

    def transcribe_file(self, audio_path: str) -> str:
        """
        Transcribe el audio REAL usando Whisper.
        """
        if self.model is None:
            self.load()

        print(f"[ASR] Transcribiendo archivo: {audio_path}")

        result = self.model.transcribe(audio_path, language="es")
        text = result.get("text", "").strip()

        if not text:
            return "(sin texto detectado)"

        return text
