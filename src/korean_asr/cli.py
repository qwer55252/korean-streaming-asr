import argparse
from collections import deque
import json
import math
import queue
import sys
import threading
import time
import wave
import numpy as np
from scipy.signal import resample_poly
from .backend import WhisperServer
from .stream import BLOCK, RATE, Segmenter

class Worker:
    """Replace obsolete partials; preserve finals with explicit overload failure."""
    def __init__(self, backend, emit):
        self.backend, self.emit = backend, emit
        self.jobs = deque()
        self.condition = threading.Condition()
        self.stopping = False
        self.error = None
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def submit(self, job):
        if job is None:
            return
        with self.condition:
            if self.error:
                raise RuntimeError(f"Inference failed: {self.error}") from self.error
            self.jobs = deque(j for j in self.jobs if j.final)
            if len(self.jobs) >= 4:
                raise RuntimeError("Inference cannot keep up. Use a smaller model or close other apps.")
            self.jobs.append(job)
            self.condition.notify()

    def run(self):
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.jobs or self.stopping)
                    if not self.jobs:
                        return
                    job = self.jobs.popleft()
                started = time.monotonic()
                text = self.backend.transcribe(job.audio)
                self.emit({'type': 'final' if job.final else 'partial',
                           'utterance': job.utterance, 'text': text,
                           'audio_end_s': round(job.end, 2),
                           'audio_duration_s': round(len(job.audio) / RATE, 2),
                           'inference_s': round(time.monotonic() - started, 3)})
        except Exception as exc:
            self.error = exc

    def close(self):
        with self.condition:
            self.stopping = True
            self.condition.notify()
        self.thread.join()
        if self.error:
            raise RuntimeError(f"Inference failed: {self.error}") from self.error

def wav_blocks(path):
    with wave.open(str(path), 'rb') as wav:
        if wav.getsampwidth() != 2:
            raise ValueError('Replay requires PCM16 WAV; convert with ffmpeg (see README).')
        rate, channels = wav.getframerate(), wav.getnchannels()
        audio = np.frombuffer(wav.readframes(wav.getnframes()), '<i2').astype(np.float32)
    audio = audio.reshape(-1, channels).mean(axis=1) / 32768
    factor = math.gcd(rate, RATE)
    audio = resample_poly(audio, RATE // factor, rate // factor)
    for i in range(0, len(audio), BLOCK):
        yield np.pad(audio[i:i + BLOCK], (0, max(0, i + BLOCK - len(audio))))

def main():
    p = argparse.ArgumentParser(description='한국어 로컬 실시간 자막 (Ctrl+C 종료)')
    p.add_argument('--server', default='http://127.0.0.1:8080')
    p.add_argument('--list-devices', action='store_true')
    p.add_argument('--device', type=int)
    p.add_argument('--wav', help='Replay a PCM16 WAV through the same streaming pipeline')
    p.add_argument('--realtime', action='store_true', help='Pace WAV input in real time')
    p.add_argument('--json', action='store_true', help='Emit JSONL events')
    p.add_argument('--threshold', type=float, default=.008, help='RMS speech gate (not neural VAD)')
    p.add_argument('--silence', type=float, default=.8)
    p.add_argument('--interval', type=float, default=1)
    p.add_argument('--max-seconds', type=float, default=12)
    args = p.parse_args()
    if args.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return
    def emit(event):
        if args.json:
            print(json.dumps(event, ensure_ascii=False), flush=True)
        else:
            label = '확정' if event['type'] == 'final' else '임시'
            print(f"[{label} #{event['utterance']}] {event['text']} ({event['inference_s']:.2f}s)", flush=True)
    worker = None
    try:
        segmenter = Segmenter(args.threshold, args.silence, args.interval, args.max_seconds)
        backend = WhisperServer(args.server)
        print('로컬 모델 서버 준비 대기 중...', file=sys.stderr)
        backend.wait_ready()
        worker = Worker(backend, emit)
        if args.wav:
            for block in wav_blocks(args.wav):
                if args.realtime:
                    time.sleep(.1)
                worker.submit(segmenter.feed(block))
                if not args.realtime:
                    # Offline smoke tests should not overload the asynchronous worker.
                    with worker.condition:
                        while worker.jobs and not worker.error:
                            worker.condition.wait(timeout=.05)
        else:
            import sounddevice as sd
            device = sd.query_devices(args.device, 'input')
            rate = int(device['default_samplerate'])
            divisor = math.gcd(rate, RATE)
            audio_queue = queue.Queue(maxsize=30)
            overflow = threading.Event()
            def callback(indata, frames, timing, status):
                if status:
                    overflow.set()
                try:
                    audio_queue.put_nowait(indata[:, 0].copy())
                except queue.Full:
                    overflow.set()
            print('마이크 시작. 종료: Ctrl+C', file=sys.stderr)
            with sd.InputStream(device=args.device, channels=1, samplerate=rate,
                                blocksize=rate // 10, dtype='float32', callback=callback):
                try:
                    while True:
                        if overflow.is_set():
                            raise RuntimeError('Audio capture overflow; close other apps and restart.')
                        if worker.error:
                            raise RuntimeError(str(worker.error))
                        try:
                            audio = audio_queue.get(timeout=.2)
                        except queue.Empty:
                            continue
                        block = resample_poly(audio, RATE // divisor, rate // divisor)
                        worker.submit(segmenter.feed(block))
                except KeyboardInterrupt:
                    pass
        worker.submit(segmenter.flush())
    except (Exception, KeyboardInterrupt) as exc:
        print(f'오류: {exc}', file=sys.stderr)
        return_code = 1
    else:
        return_code = 0
    finally:
        if worker:
            try:
                worker.close()
            except Exception as exc:
                print(f'오류: {exc}', file=sys.stderr)
                return_code = 1
    if return_code:
        raise SystemExit(return_code)
