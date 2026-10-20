import threading
import time
import numpy as np
from korean_asr.stream import BLOCK, Segmenter, Job
from korean_asr.cli import Worker, wav_blocks

VOICE = np.full(BLOCK, .05, dtype=np.float32)
QUIET = np.zeros(BLOCK, dtype=np.float32)

def test_silence_does_not_create_jobs():
    s = Segmenter()
    assert all(s.feed(QUIET) is None for _ in range(1000))
    assert s.flush() is None
    assert len(s.pre) == 3

def test_partial_then_final_and_reset():
    s = Segmenter()
    jobs = [s.feed(VOICE) for _ in range(15)]
    jobs += [s.feed(QUIET) for _ in range(8)]
    jobs = [j for j in jobs if j]
    assert any(not j.final for j in jobs)
    assert jobs[-1].final
    assert jobs[-1].utterance == 0
    assert s.flush() is None
    s.feed(VOICE)
    assert s.flush().utterance == 1

def test_continuous_speech_is_bounded():
    s = Segmenter(max_seconds=2)
    jobs = [s.feed(VOICE) for _ in range(100)]
    finals = [j for j in jobs if j and j.final]
    assert len(finals) == 5
    assert all(len(j.audio) <= 2 * 16000 for j in finals)

def test_worker_coalesces_pending_partials_and_keeps_final():
    entered, release = threading.Event(), threading.Event()
    class Backend:
        def transcribe(self, audio):
            entered.set()
            assert release.wait(3)
            return '안녕하세요'
    events = []
    w = Worker(Backend(), events.append)
    w.submit(Job(0, VOICE, False, 1))
    assert entered.wait(3)
    w.submit(Job(0, VOICE, False, 2))
    w.submit(Job(0, VOICE, True, 3))
    w.submit(Job(1, VOICE, False, 4))
    release.set()
    w.close()
    assert [(e['utterance'], e['type']) for e in events] == [(0, 'partial'), (0, 'final'), (1, 'partial')]

def test_backend_failure_is_propagated():
    import pytest
    class Broken:
        def transcribe(self, audio):
            raise ValueError('unavailable')
    w = Worker(Broken(), lambda e: None)
    w.submit(Job(0, VOICE, True, 1))
    with pytest.raises(RuntimeError, match='unavailable'):
        w.close()

def test_stereo_48k_wav_resampling(tmp_path):
    import wave
    path = tmp_path / 'audio.wav'
    with wave.open(str(path), 'wb') as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(48000)
        wav.writeframes(np.zeros(48000 * 2, dtype='<i2').tobytes())
    blocks = list(wav_blocks(path))
    assert len(blocks) == 10
    assert all(len(b) == BLOCK for b in blocks)

def test_overload_fails_instead_of_discarding_finals():
    import pytest
    entered, release = threading.Event(), threading.Event()
    class Slow:
        def transcribe(self, audio):
            entered.set()
            assert release.wait(3)
            return '음성'
    w = Worker(Slow(), lambda e: None)
    w.submit(Job(0, VOICE, True, 0))
    assert entered.wait(3)
    try:
        for i in range(1, 5):
            w.submit(Job(i, VOICE, True, i))
        with pytest.raises(RuntimeError, match='cannot keep up'):
            w.submit(Job(5, VOICE, True, 5))
    finally:
        release.set()
        w.close()

def test_http_request_is_korean_pcm16_and_no_translation():
    import io
    import wave
    from korean_asr.backend import WhisperServer
    backend = WhisperServer('http://127.0.0.1:8080')
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {'text': ' 안녕하세요 '}
    class Session:
        def post(self, url, files, timeout):
            assert files['language'] == (None, 'ko')
            assert files['translate'] == (None, 'false')
            with wave.open(io.BytesIO(files['file'][1])) as wav:
                assert wav.getframerate() == 16000
                assert wav.getsampwidth() == 2
                assert wav.getnchannels() == 1
            return Response()
    backend.session = Session()
    assert backend.transcribe(VOICE) == '안녕하세요'
