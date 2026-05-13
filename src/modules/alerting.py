import logging
import os
from typing import Optional

from twilio.rest import Client

logger = logging.getLogger(__name__)


class AlertService:
    def __init__(self) -> None:
        self.enabled = os.getenv("ALERTS_ENABLED", "false").lower() == "true"
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.from_number = os.getenv("TWILIO_FROM_NUMBER")
        self.supervisor_number = os.getenv("ALERT_SUPERVISOR_NUMBER")
        self._client: Optional[Client] = None

    def _get_client(self) -> Optional[Client]:
        if not self.enabled:
            return None
        if not (self.account_sid and self.auth_token):
            return None
        if self._client is None:
            self._client = Client(self.account_sid, self.auth_token)
        return self._client

    def send_high_risk_alert(self, final_risk: float, transcript: str) -> None:
        client = self._get_client()
        if client is None or not self.supervisor_number or not self.from_number:
            logger.warning("AlertService: configuración incompleta o desactivada. Alerta no enviada.")
            return
        message_body = (
            "Alerta de alto riesgo detectada en llamada (BERT).\n"
            f"Score: {final_risk:.3f}\n"
            f"Extracto: {transcript[:200]}..."
        )
        try:
            client.messages.create(
                body=message_body,
                from_=self.from_number,
                to=self.supervisor_number,
            )
            logger.info("Alerta de alto riesgo enviada a supervisor.")
        except Exception:
            logger.exception("Error al enviar alerta SMS via Twilio.")
