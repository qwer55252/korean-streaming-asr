"""Bounded utterance segmentation; audio blocks must be 100 ms at 16 kHz."""
from dataclasses import dataclass
import numpy as np

RATE = 16000
BLOCK = 1600

@dataclass
class Job:
    utterance: int
    audio: np.ndarray
    final: bool
    end: float

class Segmenter:
    def __init__(self, threshold=0.008, silence=0.8, interval=1.0, max_seconds=12):
        if not (threshold > 0 and silence >= .1 and interval >= .1 and max_seconds >= 2):
            raise ValueError("threshold > 0, silence/interval >= 0.1, max_seconds >= 2 required")
        self.threshold, self.silence, self.interval = threshold, silence, interval
        self.max_seconds = max_seconds
        self.pre = []
        self.frames = []
        self.quiet = self.elapsed = self.last_update = 0.0
        self.utterance = 0

    def feed(self, block):
        if len(block) != BLOCK:
            raise ValueError("Expected 100 ms mono audio")
        self.elapsed += .1
        voiced = float(np.sqrt(np.mean(block ** 2))) >= self.threshold
        if not self.frames:
            self.pre.append(block.copy())
            self.pre = self.pre[-3:]
            if not voiced:
                return None
            self.frames, self.pre = self.pre, []
            self.last_update = self.elapsed
        else:
            self.frames.append(block.copy())
        self.quiet = 0 if voiced else self.quiet + .1
        final = self.quiet + 1e-6 >= self.silence or len(self.frames) * .1 >= self.max_seconds
        if final or self.elapsed - self.last_update + 1e-6 >= self.interval:
            job = Job(self.utterance, np.concatenate(self.frames), final, self.elapsed)
            self.last_update = self.elapsed
            if final:
                self.frames = []
                self.quiet = 0
                self.utterance += 1
            return job
        return None

    def flush(self):
        if not self.frames:
            return None
        job = Job(self.utterance, np.concatenate(self.frames), True, self.elapsed)
        self.frames = []
        self.utterance += 1
        return job
