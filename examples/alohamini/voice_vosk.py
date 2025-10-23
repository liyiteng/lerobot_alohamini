# -*- coding: utf-8 -*-
"""
voice_control_vosk.py — Local streaming ASR (CPU) with Vosk for Chinese commands.

Minimal pipeline:
  sounddevice (mic) → (optional resample to 16k) → Vosk streaming → parse → on_command

Install (once):
  pip install vosk sounddevice numpy
  # 下载并解压中文模型（示例：vosk-model-small-cn-0.22），把目录填到 cfg.vosk_model_path

Usage:
  from voice_control_vosk import VoiceConfig, VoiceEngine
  cfg = VoiceConfig(vosk_model_path="/path/to/vosk-model-small-cn-0.22")
  eng = VoiceEngine(cfg, on_command=lambda c: print("→", c))
  eng.start()
"""
from __future__ import annotations

import json
import re
import time
import queue
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Tuple

import numpy as np
import sounddevice as sd

from pathlib import Path

try:
    from vosk import Model as VoskModel, KaldiRecognizer
except Exception as e:
    VoskModel = None  # type: ignore

BASE_DIR = Path(__file__).resolve().parent          # 当前脚本所在目录
MODEL_DIR = BASE_DIR / "vosk-model-cn-0.22"   # 同级目录的模型文件夹-small

# ---------------------------
# Config
# ---------------------------

@dataclass
class VoiceConfig:
    # Audio capture (we auto-fallback to supported combos)
    samplerate: int = 44_100            # preferred device SR; will fallback to device default/48k/44.1k/32k/16k
    channels: int = 1
    device_index: Optional[int] = None  # None = auto-pick an input device
    chunk_seconds: float = 0.1          # smaller window → smoother streaming
    min_dbfs: float = -30.0             # energy gate to skip noise

    # Vosk ASR
    vosk_model_path: str = "./vosk-model-cn-0.22"  # ← 修改为你的模型目录
    vosk_sample_rate: int = 16_000      # most CN models are 16k
    grammar: List[str] = field(default_factory=lambda: [
        "停止", "上升", "上升10厘米", "上升5厘米" , "降低", "前进", "后退", "向右", "向左", "右转", "左转",
        "旋转", "转", "毫米", "厘米", "度", "Z轴", "底盘", "夹爪", "开合", "归零", "关力矩"
    ])

    # Parser / dispatch
    command_cooldown_s: float = 0.8
    print_partial: bool = True


# ---------------------------
# Utils
# ---------------------------

def dbfs(x: np.ndarray) -> float:
    if x.size == 0: return -120.0
    rms = float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))
    if rms <= 1e-9: return -120.0
    return 20.0 * np.log10(min(max(rms, 1e-9), 1.0))

def list_input_devices() -> List[Tuple[int, dict]]:
    devices = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0:
            devices.append((i, d))
    return devices

def resample_linear(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Lightweight linear resampler x(float32 mono)."""
    if sr_in == sr_out or x.size == 0:
        return x.astype(np.float32, copy=False)
    n_out = int(round(x.size * sr_out / sr_in))
    if n_out <= 1:
        return x[:1].astype(np.float32, copy=False)
    xp = np.linspace(0.0, 1.0, x.size, endpoint=False, dtype=np.float32)
    xq = np.interp(np.linspace(0.0, 1.0, n_out, endpoint=False, dtype=np.float32), xp, x).astype(np.float32)
    return xq


# ---------------------------
# Command Parser
# ---------------------------

class CommandParser:
    def __init__(self) -> None:
        self.rules: List[tuple[re.Pattern, str]] = [
            (re.compile(r"(紧急|立刻)?停(止)?|别动|停止|stop", re.I), "stop"),
            (re.compile(r"(升高|上升|抬高|升一点|向上|up)", re.I), "lift_up"),
            (re.compile(r"(降低|下降|放下|降一点|向下|down)", re.I), "lift_down"),
            (re.compile(r"(?:向|像|相)?右(旋转|转)|右转|turn right", re.I), "turn_right"),
            (re.compile(r"(?:向|像|相)?左(旋转|转)|左转|turn left",  re.I), "turn_left"),
            (re.compile(r"(前进|往前|向前|go|forward)", re.I), "forward"),
            (re.compile(r"(后退|往后|向后|back|backward)", re.I), "backward"),
        ]

    def parse(self, text: str) -> Optional[str]:
        for pat, cmd in self.rules:
            if pat.search(text):
                return cmd
        return None


# ---------------------------
# Engine (Streaming with Vosk)
# ---------------------------

class VoiceEngine:
    def __init__(self, cfg: VoiceConfig, on_command: Optional[Callable[[str], None]] = None):
        if VoskModel is None:
            raise RuntimeError("vosk is not installed. pip install vosk")
        self.cfg = cfg
        self.on_command = on_command or (lambda cmd: print(f"→ 执行动作: {cmd}"))
        self.parser = CommandParser()

        self._audio_q: queue.Queue[np.ndarray] = queue.Queue()
        self._stop_evt = threading.Event()
        self._stream: Optional[sd.InputStream] = None
        self._asr_thread: Optional[threading.Thread] = None
        self._rec = None   # KaldiRecognizer
        self._last_dispatch_ts = 0.0

    def start(self) -> None:
        self._stop_evt.clear()
        self._init_recognizer()
        self._open_stream()
        self._asr_thread = threading.Thread(target=self._asr_loop, daemon=True)
        self._asr_thread.start()
        print("🎤 Vosk 流式语音已启动（Ctrl+C 停止）")

    def stop(self) -> None:
        self._stop_evt.set()
        try:
            if self._asr_thread:
                self._asr_thread.join(timeout=2.0)
        finally:
            if self._stream:
                self._stream.stop(); self._stream.close(); self._stream = None
        print("🛑 已停止")

    def _init_recognizer(self) -> None:
        print(f"⏳ 加载 Vosk 模型: {self.cfg.vosk_model_path}")
        model = VoskModel(str(self.cfg.vosk_model_path))
        if self.cfg.grammar:
            self._rec = KaldiRecognizer(model, self.cfg.vosk_sample_rate,
                                        json.dumps(self.cfg.grammar, ensure_ascii=False))
        else:
            self._rec = KaldiRecognizer(model, self.cfg.vosk_sample_rate)
        self._rec.SetWords(True)  # include word timings (optional)
        print("✅ 模型就绪")

    def _open_stream(self) -> None:
        def _cb(indata, frames, t, status):
            if status:  # limit spam
                print(status)
            mono = np.mean(indata, axis=1).astype(np.float32)
            self._audio_q.put(mono)

        # Build device candidates
        candidates_dev: List[int] = []
        if self.cfg.device_index is not None:
            candidates_dev = [self.cfg.device_index]
        else:
            try:
                default_in = sd.default.device[0]
                if isinstance(default_in, int) and default_in >= 0:
                    candidates_dev.append(default_in)
            except Exception:
                pass
            for i, _info in list_input_devices():
                if i not in candidates_dev:
                    candidates_dev.append(i)
        if not candidates_dev:
            raise RuntimeError("No input-capable devices found.")

        last_err = None
        for dev in candidates_dev:
            try:
                d_info = sd.query_devices(dev, 'input')
            except Exception as e:
                last_err = e
                continue

            # sample rate candidates
            sr_cand: List[int] = []
            if self.cfg.samplerate:
                sr_cand.append(int(self.cfg.samplerate))
            dflt_sr = int(d_info.get("default_samplerate") or 0)
            if dflt_sr and dflt_sr not in sr_cand:
                sr_cand.append(dflt_sr)
            for sr in [48000, 44100, 32000, 16000, 22050]:
                if sr not in sr_cand:
                    sr_cand.append(sr)

            # channel candidates
            ch_cand: List[int] = []
            if self.cfg.channels:
                ch_cand.append(int(self.cfg.channels))
            if 1 not in ch_cand:
                ch_cand.append(1)
            if int(d_info.get("max_input_channels", 0)) >= 2 and 2 not in ch_cand:
                ch_cand.append(2)

            for sr in sr_cand:
                for ch in ch_cand:
                    try:
                        stream = sd.InputStream(
                            samplerate=sr,
                            channels=ch,
                            dtype="int16",
                            device=dev,
                            callback=_cb,
                            blocksize=max(128, int(sr * self.cfg.chunk_seconds)),
                        )
                        stream.start()
                        # success
                        self._stream = stream
                        if sr != self.cfg.samplerate:
                            print(f"⚠️  采样率 {self.cfg.samplerate} 不可用，已改用 {sr}")
                            self.cfg.samplerate = sr
                        if ch != self.cfg.channels:
                            print(f"⚠️  通道数 {self.cfg.channels} 不可用，已改用 {ch}")
                            self.cfg.channels = ch
                        if dev != self.cfg.device_index:
                            name = d_info.get('name', str(dev))
                            print(f"✅  使用设备 #{dev}: {name}")
                            self.cfg.device_index = dev
                        print(f"ℹ️  采样率={sr}, 通道={ch}, block={max(128, int(sr * self.cfg.chunk_seconds))} 帧")
                        return
                    except sd.PortAudioError as e:
                        last_err = e
                        continue

        raise last_err or RuntimeError("Failed to open audio input stream")

    def _asr_loop(self) -> None:
        sr_in = self.cfg.samplerate
        sr_out = self.cfg.vosk_sample_rate
        chunk_len = int(sr_in * self.cfg.chunk_seconds)
        buf = np.zeros(0, dtype=np.float32)

        while not self._stop_evt.is_set():
            try:
                x = self._audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            buf = np.concatenate([buf, x])
            if buf.size < chunk_len:
                continue

            clip = buf[:chunk_len]
            buf = buf[chunk_len:]

            # energy gate
            level = dbfs(clip)
            if level < self.cfg.min_dbfs:
                continue

            # resample to model SR & int16 bytes
            y = resample_linear(clip, sr_in, sr_out)
            y_i16 = np.clip(y, -1.0, 1.0)
            y_i16 = (y_i16 * 32767.0).astype(np.int16).tobytes()

            # streaming accept
            if self._rec.AcceptWaveform(y_i16):
                # final result
                result = json.loads(self._rec.Result() or "{}")
                text = (result.get("text") or "").strip()
                if text:
                    if self.cfg.print_partial:
                        print(f"⏹ 句末:", text, f"({level:.1f} dBFS)")
                    self._trigger_if_match(text)
            else:
                # partial result
                if self.cfg.print_partial:
                    part = json.loads(self._rec.PartialResult() or "{}").get("partial", "")
                    if part:
                        print(time.strftime("[%H:%M:%S] "), part, f"({level:.1f} dBFS)")

    def _trigger_if_match(self, text: str) -> None:
        now = time.time()
        cmd = self.parser.parse(text)
        if cmd and (now - self._last_dispatch_ts) >= self.cfg.command_cooldown_s:
            self._last_dispatch_ts = now
            try:
                self.on_command(cmd)
            except Exception as e:
                print("❌ on_command 异常：", e)


# ---------------------------
# Demo
# ---------------------------

if __name__ == "__main__":
    def demo_on_command(cmd: str) -> None:
        print("→ 执行动作:", cmd)

    cfg = VoiceConfig(
        samplerate=44100,                  # 设备 SR；会重采样到 16k 送给 Vosk
        device_index=None,                 # 自动选择输入设备
        vosk_model_path=MODEL_DIR,  # 改成你的模型解压目录
        chunk_seconds=0.5,
        min_dbfs=-30.0,
        print_partial=True,
    )
    eng = VoiceEngine(cfg, on_command=demo_on_command)

    try:
        eng.start()
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        eng.stop()
