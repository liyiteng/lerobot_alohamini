#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpeechEngineGummy — ASR-only engine (mirrors voice_gummy.py behavior but outputs *text only*).
Key parity with your voice_gummy.py:
- Mic open at device rate with robust fallback; online resample → 16k
- Energy gating with env tracking, relative margin, periodic volume log
- Phrase segmentation: end-silence + max phrase length
- DashScope Gummy one-shot sessions per phrase
- Hotword vocabulary create/update with graceful fallback
- Non-blocking get_text_nowait() returning finalized utterances
"""

from __future__ import annotations
import os
import re
import time
import math
import queue
import threading
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np
import sounddevice as sd

# ---------- optional high-quality resampling ----------
try:
    from scipy.signal import resample_poly  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

# ---------- DashScope (Gummy) ----------
import dashscope
from dashscope.audio.asr import (
    TranslationRecognizerChat,
    TranslationRecognizerCallback,
    TranscriptionResult,
    TranslationResult,
)

# ---------- Vocabulary service (best-effort) ----------
_VOCAB_AVAILABLE = True
try:
    from dashscope.audio.asr import VocabularyService
except Exception:
    _VOCAB_AVAILABLE = False
    VocabularyService = None  # type: ignore


# ===================== utils =====================
def dbfs(x: np.ndarray) -> float:
    eps = 1e-12
    rms = max(eps, float(np.sqrt(np.mean(np.square(x.astype(np.float64))))))
    return 20.0 * math.log10(rms + eps)


def resample_to_16k(x: np.ndarray, src_sr: int) -> np.ndarray:
    if src_sr == 16000 or len(x) == 0:
        return x.astype(np.float32, copy=False)
    if _HAS_SCIPY:
        from math import gcd
        g = gcd(src_sr, 16000)
        up, down = 16000 // g, src_sr // g
        y = resample_poly(x.astype(np.float32, copy=False), up, down)
        return np.clip(y, -1.0, 1.0).astype(np.float32, copy=False)
    new_len = int(round(len(x) * (16000.0 / float(src_sr))))
    if new_len <= 1:
        return np.zeros(0, dtype=np.float32)
    xp = np.linspace(0.0, 1.0, num=len(x), endpoint=False, dtype=np.float64)
    xnew = np.linspace(0.0, 1.0, num=new_len, endpoint=False, dtype=np.float64)
    y = np.interp(xnew, xp, x.astype(np.float64))
    return y.astype(np.float32, copy=False)


def float32_to_pcm16(x: np.ndarray) -> bytes:
    x = np.clip(x, -1.0, 1.0)
    return (x * 32767.0).astype(np.int16).tobytes()


# ===================== config =====================
@dataclass
class SpeechConfig:
    # local audio
    target_sr: int = 16000
    channels: int = 1
    chunk_seconds: float = 0.05
    overlap_seconds: float = 0.01  # keep tail for continuity

    # energy gate
    min_dbfs: float = -30.0
    rel_db_margin_db: float = 7.0
    env_track_alpha: float = 0.9
    verbose_vol: bool = True

    # segmentation
    speech_end_silence_ms: int = 1000
    max_phrase_seconds: float = 15.0

    # Gummy
    model: str = "gummy-chat-v1"
    gummy_max_end_silence_ms: int = 1200
    print_partial: bool = False

    # Hotwords
    vocabulary_prefix: Optional[str] = None
    hotwords: Optional[List[str]] = field(default_factory=list)

    # diagnostics
    verbose_open: bool = True


# ===================== vocabulary manager (best-effort) =====================
def ensure_vocabulary_id(prefix: Optional[str], words: Optional[List[str]], target_model: str) -> Optional[str]:
    if not prefix or not words:
        return None
    if not _VOCAB_AVAILABLE:
        print("⚠️ dashscope without VocabularyService; skipping vocabulary.")
        return None
    try:
        svc = VocabularyService()
        # try update existing with same prefix (best-effort)
        try:
            ex = svc.list_vocabularies(prefix=prefix, page_index=0, page_size=10)
            if isinstance(ex, list) and ex:
                vid = ex[0].get("vocabulary_id") or ex[0].get("id")
                if vid:
                    vocab = []
                    for w in words:
                        if not isinstance(w, str) or not w.strip(): continue
                        if any('\u4e00' <= c <= '\u9fff' for c in w):
                            vocab.append({"text": w, "lang": "zh"})
                        else:
                            vocab.append({"text": w, "lang": "en"})
                    svc.update_vocabulary(vid, vocab)
                    print(f"✓ updated vocabulary: {vid}")
                    return vid
        except Exception:
            pass
        # make new
        vocab = []
        for w in words:
            if not isinstance(w, str) or not w.strip(): continue
            if any('\u4e00' <= c <= '\u9fff' for c in w):
                vocab.append({"text": w, "lang": "zh"})
            else:
                vocab.append({"text": w, "lang": "en"})
        res = svc.create_vocabulary(target_model=target_model, prefix=prefix[:9], vocabulary=vocab)
        if isinstance(res, dict):
            vid = res.get("vocabulary_id") or res.get("id") or res.get("output", {}).get("vocabulary_id")
        else:
            vid = str(res)
        if vid:
            print(f"✓ created vocabulary: {vid}")
            return vid
    except Exception as e:
        print(f"⚠️ vocabulary failed: {e}")
    return None


# ===================== one-shot Gummy chat =====================
class _GummyOneShot(TranslationRecognizerCallback):
    def __init__(self, cfg: SpeechConfig, vocabulary_id: Optional[str] = None):
        self.cfg = cfg
        self._vid = vocabulary_id
        self._cli: Optional[TranslationRecognizerChat] = None
        self._final_text: str = ""
        self._opened = threading.Event()
        self._closed = threading.Event()
        self._lock = threading.Lock()
        self._err: Optional[str] = None

    def on_open(self):
        self._opened.set()

    def on_event(self, request_id, transcription_result: TranscriptionResult, translation_result: TranslationResult, usage):
        if transcription_result is not None and transcription_result.text:
            with self._lock:
                self._final_text = transcription_result.text

    def on_error(self, result):
        with self._lock:
            self._err = f"Gummy error: {result}"
        self._closed.set()

    def on_complete(self):
        self._closed.set()

    def on_close(self):
        self._closed.set()

    def start(self):
        self._cli = TranslationRecognizerChat(
            model=self.cfg.model,
            format="pcm",
            sample_rate=16000,
            transcription_enabled=True,
            callback=self,
            max_end_silence=self.cfg.gummy_max_end_silence_ms,
            vocabulary_id=self._vid if self._vid else None,
        )
        self._cli.start()
        self._opened.wait(timeout=5.0)

    def send_audio(self, pcm_bytes: bytes) -> bool:
        if not self._cli:
            return False
        return self._cli.send_audio_frame(pcm_bytes)

    def stop(self):
        if self._cli:
            self._cli.stop()
        self._closed.wait(timeout=5.0)

    @property
    def final_text(self) -> str:
        with self._lock:
            return (self._final_text or "").strip()

    @property
    def error(self) -> Optional[str]:
        with self._lock:
            return self._err


# ===================== engine =====================
class SpeechEngineGummy:
    """
    Produces finalized texts based on mic audio and Gummy ASR.
    No command parsing here.
    """
    def __init__(self, cfg: SpeechConfig):
        self.cfg = cfg
        api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Please set env DASHSCOPE_API_KEY.")
        dashscope.api_key = api_key

        self._q_audio: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=64)
        self._q_text: "queue.Queue[str]" = queue.Queue(maxsize=32)
        self._stop_evt = threading.Event()
        self._worker: Optional[threading.Thread] = None

        self._env_db: Optional[float] = None
        self._last_vol_print = 0.0

        self._speech_active = False
        self._cloud: Optional[_GummyOneShot] = None
        self._last_voice_ts = 0.0
        self._phrase_start: Optional[float] = None

        self.device_sr = 16000
        self._stream: Optional[sd.InputStream] = None

        # vocabulary best-effort
        self._vocabulary_id: Optional[str] = None
        if self.cfg.vocabulary_prefix and self.cfg.hotwords:
            self._vocabulary_id = ensure_vocabulary_id(self.cfg.vocabulary_prefix, self.cfg.hotwords, self.cfg.model)

    # audio callback
    def _audio_cb(self, indata: np.ndarray, frames: int, time_info, status):
        mono = indata[:, 0].copy()
        try:
            self._q_audio.put_nowait(mono)
        except queue.Full:
            pass

    def start(self):
        # open mic: prefer device default (e.g. 44.1k/48k), we will resample to 16k
        try:
            self._stream = sd.InputStream(
                samplerate=None, channels=self.cfg.channels, dtype="float32",
                blocksize=int(self.cfg.chunk_seconds * 16000),
                callback=self._audio_cb,
            )
            self._stream.start()
            self.device_sr = int(round(self._stream.samplerate))
        except Exception:
            self._stream = sd.InputStream(
                samplerate=16000, channels=self.cfg.channels, dtype="float32",
                blocksize=int(self.cfg.chunk_seconds * 16000),
                callback=self._audio_cb,
            )
            self._stream.start()
            self.device_sr = 16000

        # start first cloud session
        self._cloud = _GummyOneShot(self.cfg, self._vocabulary_id)
        self._cloud.start()

        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

        vid_str = f", vocab_id={self._vocabulary_id}" if self._vocabulary_id else ""
        print(f"[ASR] opened: model={self.cfg.model}{vid_str}")
        print(f"🎤 SpeechEngineGummy started: device_sr={self.device_sr}Hz -> 16k")

    def stop(self):
        self._stop_evt.set()
        if self._worker:
            self._worker.join(timeout=2.0)
        if self._cloud:
            self._cloud.stop()
        if self._stream:
            self._stream.stop(); self._stream.close()
        print("🎤 SpeechEngineGummy stopped.")

    def _run(self):
        chunk = max(1, int(self.device_sr * self.cfg.chunk_seconds))
        tail = int(self.device_sr * self.cfg.overlap_seconds)
        silence_needed = self.cfg.speech_end_silence_ms / 1000.0
        buf = np.zeros(0, dtype=np.float32)

        while not self._stop_evt.is_set():
            try:
                piece = self._q_audio.get(timeout=0.2)
                buf = np.concatenate([buf, piece])
            except queue.Empty:
                pass

            while len(buf) >= chunk:
                clip = buf[:chunk]
                buf = buf[chunk - tail:] if tail > 0 else buf[chunk:]

                sig16 = resample_to_16k(clip, self.device_sr)
                level = dbfs(sig16)

                if self._env_db is None or (level < self.cfg.min_dbfs):
                    a = self.cfg.env_track_alpha
                    self._env_db = level if self._env_db is None else (a*self._env_db + (1.0-a)*level)

                now = time.time()
                if self.cfg.verbose_vol and (now - self._last_vol_print >= 1.0):
                    env = self._env_db if self._env_db is not None else level
                    thr = max(self.cfg.min_dbfs, (env if env else level) + self.cfg.rel_db_margin_db)
                    print(f"[VOL] frame {level:.1f} dBFS | env {env:.1f} dBFS | gate >= {thr:.1f}")
                    self._last_vol_print = now

                env = self._env_db if self._env_db is not None else level
                rel_gate = (env if env is not None else level) + self.cfg.rel_db_margin_db
                gate = max(self.cfg.min_dbfs, rel_gate)
                is_voice = (level >= gate)

                if is_voice:
                    if not self._speech_active:
                        self._speech_active = True
                        self._phrase_start = now
                        if self._cloud and self._cloud.error:
                            self._cloud = _GummyOneShot(self.cfg, self._vocabulary_id)
                            self._cloud.start()
                    self._last_voice_ts = now
                    if self._cloud:
                        self._cloud.send_audio(float32_to_pcm16(sig16))
                else:
                    if self._speech_active and (now - self._last_voice_ts) >= silence_needed:
                        # phrase ended
                        self._speech_active = False
                        if self._cloud:
                            self._cloud.stop()
                            txt = self._cloud.final_text
                            err = self._cloud.error
                            self._cloud = None
                            if not err and txt:
                                try:
                                    self._q_text.put_nowait(txt)
                                except queue.Full:
                                    pass
                        self._cloud = _GummyOneShot(self.cfg, self._vocabulary_id)
                        self._cloud.start()

                # hard cut long phrase
                if self._speech_active and self._phrase_start and (now - self._phrase_start) > self.cfg.max_phrase_seconds:
                    self._speech_active = False
                    if self._cloud:
                        self._cloud.stop()
                        txt = self._cloud.final_text
                        err = self._cloud.error
                        self._cloud = None
                        if not err and txt:
                            try:
                                self._q_text.put_nowait(txt)
                            except queue.Full:
                                pass
                    self._cloud = _GummyOneShot(self.cfg, self._vocabulary_id)
                    self._cloud.start()

    # public
    def get_text_nowait(self) -> str:
        try:
            return self._q_text.get_nowait()
        except queue.Empty:
            return ""