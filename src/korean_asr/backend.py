import io
import time
import wave
import requests
import numpy as np
from .stream import RATE

class WhisperServer:
    def __init__(self, url):
        self.url = url.rstrip('/') + '/inference'
        self.session = requests.Session()

    def wait_ready(self, timeout=120):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                response = self.session.get(self.url.rsplit('/', 1)[0] + '/health', timeout=2)
                if response.ok and response.json().get('status') == 'ok':
                    return
            except (requests.RequestException, ValueError):
                pass
            time.sleep(.5)
        raise RuntimeError('Server not ready. Run python scripts/server.py in another terminal.')

    def transcribe(self, audio):
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(RATE)
            wav.writeframes((np.clip(audio, -1, 1) * 32767).astype('<i2').tobytes())
        fields = {k: (None, v) for k, v in {
            'language': 'ko', 'response_format': 'json', 'temperature': '0',
            'temperature_inc': '0', 'no_context': 'true', 'translate': 'false',
        }.items()}
        fields['file'] = ('audio.wav', buf.getvalue(), 'audio/wav')
        response = self.session.post(self.url, files=fields, timeout=(5, 90))
        response.raise_for_status()
        return response.json()['text'].strip()
