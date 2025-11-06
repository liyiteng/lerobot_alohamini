#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VoiceExecutor — text -> action
- Sticky Z target: "lift_axis.height_mm" maintained until within epsilon, with cancel phrases.
- "hold N seconds" for base motion (fuzzy).
- Relative lift delta -> absolute using update_height_mm().
- Emergency stop.
- Chinese/English number normalization.
- NEW: Built-in replay launcher on trigger phrase ("锤他" / "chui ta" / "hammer him") with cooldown.
"""

from __future__ import annotations
import re
import time
import subprocess
import sys
import shlex
from dataclasses import dataclass
from typing import Dict, Any, Optional

# ---------- config ----------
@dataclass
class ExecConfig:
    # Motion scales
    xy_speed_cmd: float = 0.20
    theta_speed_cmd: float = 500.0
    emit_text_cmd: bool = True
    z_epsilon_mm: float = 0.8  # sticky height considered reached within this tolerance

    # --- NEW: replay execution knobs ---
    enable_replay_execute: bool = True          # if True, VoiceExecutor launches replay internally
    replay_cooldown_s: float = 2.0              # avoid multiple triggers within cooldown (seconds)
    replay_default_dataset: str = "liyitenga/record_20251015131957"
    replay_default_episode: int = 0
    # command template; {python}, {dataset}, {episode} will be formatted
    replay_cmd_template: str = "{python} -m lerobot.tools.replay --dataset {dataset} --episode {episode}"
    replay_dry_run: bool = False                # if True, print command without executing (for debugging)

# ---------- units and number parsing ----------
_UNIT_MM = {
    # zh
    "毫米": 1.0, "厘米": 10.0, "米": 1000.0,
    # en
    "mm": 1.0, "millimeter": 1.0, "millimeters": 1.0,
    "cm": 10.0, "centimeter": 10.0, "centimeters": 10.0,
    "m": 1000.0, "meter": 1000.0, "meters": 1000.0,
}

_EN_UNITS = {"zero":0,"oh":0,"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9}
_EN_TEENS = {"ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,"seventeen":17,"eighteen":18,"nineteen":19}
_EN_TENS = {"twenty":20,"thirty":30,"forty":40,"fifty":50,"sixty":60,"seventy":70,"eighty":80,"ninety":90}
_EN_NUM_WORD = (
    r"(?:zero|oh|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|half|quarter)"
)
_NUM_PAT_ENFREE = rf"(?:{_EN_NUM_WORD}(?:[-\s]{_EN_NUM_WORD}){{0,4}})"
_CN_DIG = {"零":0,"〇":0,"○":0,"一":1,"二":2,"两":2,"三":3,"四":4,"五":5,"六":6,"七":7,"八":8,"九":9}

def _en_to_float(tok: str) -> Optional[float]:
    t = (tok or "").strip().lower()
    if not t: return None
    t = t.replace("-", " ")
    if t in ("half", "a half", "half a"): return 0.5
    if t in ("quarter", "a quarter"): return 0.25
    if " point " in t:
        left, right = t.split(" point ", 1)
        iv = _en_to_float(left)
        if iv is None: return None
        frac = 0.0; mul = 0.1
        for w in right.split():
            if w in _EN_UNITS: frac += _EN_UNITS[w]*mul; mul *= 0.1
            elif w in ("zero","oh"): mul *= 0.1
            else: return None
        return iv + frac
    if t.endswith(" and a half"):
        base = _en_to_float(t[: -len(" and a half")]); return (base + 0.5) if base is not None else None
    if t.endswith(" and a quarter"):
        base = _en_to_float(t[: -len(" and a quarter")]); return (base + 0.25) if base is not None else None
    parts = [w for w in t.split() if w not in ("and",)]
    if not parts: return None
    total = 0; current = 0; i = 0
    while i < len(parts):
        w = parts[i]
        if w in _EN_UNITS: current += _EN_UNITS[w]
        elif w in _EN_TEENS: current += _EN_TEENS[w]
        elif w in _EN_TENS:
            val = _EN_TENS[w]
            if i + 1 < len(parts) and parts[i+1] in _EN_UNITS:
                val += _EN_UNITS[parts[i+1]]; i += 1
            current += val
        elif w == "hundred":
            current = 100 if current == 0 else current * 100
        else: return None
        i += 1
    total += current
    if total == 0 and t in ("zero","oh"): return 0.0
    return float(total) if total != 0 else None

def _cn_to_float(tok: str) -> Optional[float]:
    tok = (tok or "").strip()
    if not tok: return None
    try: return float(tok)
    except Exception: pass
    if "点" in tok:
        left, right = tok.split("点", 1)
        lv = _cn_to_float(left) if left else 0.0
        rv = 0.0; base = 0.1
        for ch in right:
            d = _CN_DIG.get(ch, None)
            if d is None: return None
            rv += d*base; base *= 0.1
        return (lv or 0.0) + rv
    if "十" in tok:
        parts = tok.split("十")
        tens = _CN_DIG.get(parts[0], 1) if parts[0] else 1
        units = _CN_DIG.get(parts[1], 0) if len(parts) > 1 else 0
        return float(tens*10 + units)
    if tok == "半": return 0.5
    if all(ch in _CN_DIG for ch in tok):
        val = 0
        for ch in tok: val = val*10 + _CN_DIG[ch]
        return float(val)
    return None

def normalize_number(text: str) -> Optional[float]:
    if not text: return None
    t = text.lower()
    m = re.search(r"([-+]?\d+(?:\.\d+)?)", t)
    if m:
        try: return float(m.group(1))
        except Exception: pass
    m = re.search(r"[零〇○一二两三四五六七八九十点半]+", text)
    if m:
        v = _cn_to_float(m.group(0))
        if v is not None: return v
    cands = list(re.finditer(_NUM_PAT_ENFREE, t))
    if cands:
        cands.sort(key=lambda mm: (mm.start(), -(mm.end()-mm.start())), reverse=True)
        for mm in cands:
            v = _en_to_float(mm.group(0))
            if v is not None: return v
    return None

# ---------- fuzzy "hold N seconds" ----------
_EN_SEC = r"(?:seconds?|sec|s)\b"
_CN_SEC = r"(?:秒钟|秒)\b"
_SEC_ANY = fr"(?:{_CN_SEC}|{_EN_SEC})"
_NUM_PAT = rf"([-+]?\d+(?:\.\d+)?|[零〇○一二两三四五六七八九十点半]+|{_NUM_PAT_ENFREE})"

def _extract_secs_anywhere(s: str) -> Optional[float]:
    t = (s or "").lower()
    m = re.search(fr"{_NUM_PAT}[\s,.;:-]*{_SEC_ANY}", t)
    if m:
        v = normalize_number(m.group(1))
        if v is not None: return max(0.1, float(v))
    m = re.search(fr"{_SEC_ANY}[\s,.;:-]*{_NUM_PAT}", t)
    if m:
        v = normalize_number(m.group(1))
        if v is not None: return max(0.1, float(v))
    return None

def _parse_hold(s: str) -> Optional[Dict[str, Any]]:
    s = (s or "").strip().lower()
    secs = _extract_secs_anywhere(s)
    if secs is None: return None
    if any(k in s for k in ["左转","向左转","turn left","rotate left"]):  return {"kind":"rotate_left","secs":secs}
    if any(k in s for k in ["右转","向右转","turn right","rotate right"]): return {"kind":"rotate_right","secs":secs}
    if any(k in s for k in ["左移","向左平移","move left","strafe left"]):  return {"kind":"left","secs":secs}
    if any(k in s for k in ["右移","向右平移","move right","strafe right"]): return {"kind":"right","secs":secs}
    if any(k in s for k in ["前进","向前","forward","go forward","ahead"]):  return {"kind":"forward","secs":secs}
    if any(k in s for k in ["后退","向后","倒退","back","backward","go back"]): return {"kind":"backward","secs":secs}
    return None

def _kind_to_cmd(kind: str, cfg: ExecConfig) -> Dict[str, float]:
    v = cfg.xy_speed_cmd; w = cfg.theta_speed_cmd
    if kind == "forward":      return {"x.vel": +v, "y.vel": 0.0, "theta.vel": 0.0}
    if kind == "backward":     return {"x.vel": -v, "y.vel": 0.0, "theta.vel": 0.0}
    if kind == "left":         return {"x.vel": 0.0, "y.vel": +v, "theta.vel": 0.0}
    if kind == "right":        return {"x.vel": 0.0, "y.vel": -v, "theta.vel": 0.0}
    if kind == "rotate_left":  return {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": +w}
    if kind == "rotate_right": return {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": -w}
    return {}

# ---------- instant / height parsing ----------
# Sticky absolute height phrases:
#  zh: 设为/设置为/到/升到/抬到/升至/抬至  NUM(单位)
#  en: set (height|z) to NUM (mm|cm|m), set to NUM mm, raise to NUM mm, lower to NUM mm
_ABS_Z_PAT = re.compile(
    r"(?:设为|设置为|升到|抬到|升至|抬至|到|set(?:\s+(?:the\s+)?)?(?:height|z)?\s*to|raise\s+to|lower\s+to)\s*"
    r"(?P<num>[-+]?\d+(?:\.\d+)?|[零〇○一二两三四五六七八九十点半]+|"+_NUM_PAT_ENFREE+r")\s*"
    r"(?P<unit>毫米|厘米|米|mm|millimeters?|cm|centimeters?|m|meters?)?",
    re.IGNORECASE
)

# Cancel sticky Z phrases:
_CANCEL_Z_PAT = re.compile(r"(?:取消高度|取消z|取消粘滞|清除高度|停止高度|cancel\s+(?:height|z)|clear\s+(?:height|z))", re.IGNORECASE)

def parse_command(s: str) -> Dict[str, Any]:
    """
    Parse a free-form text command into an action dict.
    NOTE: For replay trigger, we set a special key '__replay' which the executor can
    choose to launch internally (preferred) or pass back to caller (legacy).
    """
    s = (s or "").strip().lower()
    out: Dict[str, Any] = {}
    if any(k in s for k in ["停止","急停","stop","停"]): return {"__stop": True}

    # absolute sticky Z target
    mz = _ABS_Z_PAT.search(s)
    if mz:
        num = mz.group("num")
        unit = mz.group("unit") or "毫米"
        val = normalize_number(num)
        if val is not None:
            mm = float(val) * _UNIT_MM.get(unit, 1.0)
            out["__sticky_z_mm"] = mm
            return out

    # cancel sticky Z
    if _CANCEL_Z_PAT.search(s):
        out["__cancel_z"] = True
        return out

    # rotation (instant)
    if any(k in s for k in ["左转","向左转","turn left","rotate left"]):
        n = normalize_number(s); out["theta.vel"] = +abs(n) if n is not None else 0.0
    if any(k in s for k in ["右转","向右转","turn right","rotate right"]):
        n = normalize_number(s); out["theta.vel"] = -abs(n) if n is not None else 0.0

    # translation
    if any(k in s for k in ["前进","向前","forward","go forward","ahead"]):
        n = normalize_number(s); unit = next((u for u in _UNIT_MM if u in s), None)
        out["x.vel"] = + (n * _UNIT_MM[unit]) / 1000.0 if unit and n is not None else +0.0
    if any(k in s for k in ["后退","向后","倒退","back","backward","go back"]):
        n = normalize_number(s); unit = next((u for u in _UNIT_MM if u in s), None)
        out["x.vel"] = - (n * _UNIT_MM[unit]) / 1000.0 if unit and n is not None else -0.0
    if any(k in s for k in ["左移","向左平移","move left","strafe left"]):
        n = normalize_number(s); unit = next((u for u in _UNIT_MM if u in s), None)
        out["y.vel"] = + (n * _UNIT_MM[unit]) / 1000.0 if unit and n is not None else +0.0
    if any(k in s for k in ["右移","向右平移","move right","strafe right"]):
        n = normalize_number(s); unit = next((u for u in _UNIT_MM if u in s), None)
        out["y.vel"] = - (n * _UNIT_MM[unit]) / 1000.0 if unit and n is not None else -0.0

    # lift (relative; executor will convert to absolute one-shot)
    if any(k in s for k in ["上升","升高","上移","up","raise","lift up"]):
        n = normalize_number(s) or 0.0; unit = next((u for u in _UNIT_MM if u in s), "毫米")
        out["lift_axis.height_mm"] = + (n * _UNIT_MM[unit])
    if any(k in s for k in ["下降","降低","下移","down","lower"]):
        n = normalize_number(s) or 0.0; unit = next((u for u in _UNIT_MM if u in s), "毫米")
        out["lift_axis.height_mm"] = - (n * _UNIT_MM[unit])

    # replay hook (let executor decide how to launch)
    if ("锤他" in s) or ("chui ta" in s) or ("hammer him" in s):
        out["__replay"] = {"dataset": "liyitenga/record_20251015131957", "episode": 0}
    return out


# ===================== executor =====================
class VoiceExecutor:
    def __init__(self, cfg: ExecConfig):
        self.cfg = cfg
        self._now_height_mm: float = 0.0
        self._held_cmd: Dict[str, float] = {}
        self._hold_until: float = 0.0
        self._one_shot_action: Dict[str, float] = {}
        self._sticky_z_target_mm: Optional[float] = None
        # NEW: last replay timestamp to enforce cooldown
        self._last_replay_ts: float = 0.0

    def update_height_mm(self, h: float):
        """Update current measured Z height (mm)."""
        self._now_height_mm = float(h)

    def get_action_nowait(self) -> Dict[str, float]:
        """Return one frame of action; includes held base command and sticky Z pursuit."""
        now = time.time()
        act: Dict[str, float] = {}
        # held base command
        if self._held_cmd and now < self._hold_until:
            act.update(self._held_cmd)
        elif self._held_cmd and now >= self._hold_until:
            self._held_cmd.clear(); self._hold_until = 0.0
        # sticky Z target
        if self._sticky_z_target_mm is not None:
            if abs(self._now_height_mm - self._sticky_z_target_mm) <= self.cfg.z_epsilon_mm:
                # reached
                self._sticky_z_target_mm = None
            else:
                act["lift_axis.height_mm"] = float(self._sticky_z_target_mm)
        # one-shot
        if self._one_shot_action:
            act.update(self._one_shot_action); self._one_shot_action.clear()
        return act

    def handle_text(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Consume a recognized text command and return an (optional) debug dict.
        Side effects:
        - may enqueue one-shot actions
        - may start 'held' actions with a deadline
        - may start sticky Z target pursuit
        - NEW: may launch a replay process internally
        """
        text = (text or "").strip()
        if not text:
            return None
        print(f"[ASR] {text}")

        # 1) HOLD pattern (time-based base motion)
        hold = _parse_hold(text)
        if hold is not None:
            kind = hold["kind"]; secs = float(hold["secs"])
            cmd = _kind_to_cmd(kind, self.cfg)
            self._held_cmd = dict(cmd)
            self._hold_until = time.time() + secs
            self._one_shot_action = dict(cmd)
            if self.cfg.emit_text_cmd:
                print(f"{cmd} for {secs:.1f}s")
            return {"__hold": secs, **cmd}

        # 2) Generic parsing (including sticky Z, stop, and replay hook)
        parsed = parse_command(text)


        if "__replay" in parsed:
            params = parsed["__replay"] or {}
            dataset = str(params.get("dataset", "liyitenga/record_20251015131957"))
            episode = int(params.get("episode", 0))
            import sys, subprocess, shlex
            cmd = [sys.executable, "examples/alohamini/replay_bi.py",
                "--dataset", dataset, "--episode", str(episode)]
            print(f"[ASR] Trigger detected → Executing: {' '.join(shlex.quote(c) for c in cmd)}")
            subprocess.Popen(cmd, cwd="/home/worker/lerobot_alohamini") 
            


        # 2b) Cancel sticky Z
        if parsed.get("__cancel_z"):
            self._sticky_z_target_mm = None
            print("✳️ sticky Z cleared")
            return {"__cancel_z": True}

        # 2c) Emergency stop
        if parsed.get("__stop"):
            self._held_cmd.clear(); self._hold_until = 0.0
            base_cmd = {"x.vel": 0.0, "y.vel": 0.0, "theta.vel": 0.0}
            z_cmd = {"lift_axis.height_mm": self._now_height_mm}
            if self.cfg.emit_text_cmd:
                print({**base_cmd, **z_cmd})
            self._one_shot_action = dict(base_cmd)
            return {"__stop": True}

        # 2d) Sticky absolute Z target
        if "__sticky_z_mm" in parsed:
            self._sticky_z_target_mm = float(parsed["__sticky_z_mm"])
            print(f"✳️ sticky Z target → {self._sticky_z_target_mm:.1f} mm (ε={self.cfg.z_epsilon_mm}mm)")
            return {"lift_axis.height_mm": self._sticky_z_target_mm}

        # 2e) Relative lift -> absolute sticky Z
        if "lift_axis.height_mm" in parsed:
            delta = float(parsed["lift_axis.height_mm"])
            self._sticky_z_target_mm = self._now_height_mm + delta
            print(f"✳️ sticky Z target (relative) → {self._sticky_z_target_mm:.1f} mm (ε={self.cfg.z_epsilon_mm}mm)")
            return {"lift_axis.height_mm": self._sticky_z_target_mm}

        # 2f) Fallback velocities (if user said "turn left/right" without number)
        if "theta.vel" in parsed and parsed["theta.vel"] == 0.0:
            parsed["theta.vel"] = self.cfg.theta_speed_cmd * (
                1.0 if ("turn left" in text.lower() or "左转" in text) else
                -1.0 if ("turn right" in text.lower() or "右转" in text) else 1.0
            )
        if "x.vel" in parsed and parsed["x.vel"] == 0.0:
            if any(k in text.lower() for k in ["前进","向前","forward","go forward","ahead"]):
                parsed["x.vel"] = +self.cfg.xy_speed_cmd
            elif any(k in text.lower() for k in ["后退","向后","倒退","back","backward","go back"]):
                parsed["x.vel"] = -self.cfg.xy_speed_cmd
        if "y.vel" in parsed and parsed["y.vel"] == 0.0:
            if any(k in text.lower() for k in ["左移","向左平移","move left","strafe left"]):
                parsed["y.vel"] = +self.cfg.xy_speed_cmd
            elif any(k in text.lower() for k in ["右移","向右平移","move right","strafe right"]):
                parsed["y.vel"] = -self.cfg.xy_speed_cmd

        # 2g) Enqueue one-shot base / Z actions
        base_cmd = {k: float(parsed[k]) for k in ("x.vel","y.vel","theta.vel") if k in parsed}
        z_cmd = {"lift_axis.height_mm": float(parsed["lift_axis.height_mm"])} if "lift_axis.height_mm" in parsed else {}
        self._one_shot_action.clear(); self._one_shot_action.update(base_cmd); self._one_shot_action.update(z_cmd)

        if self.cfg.emit_text_cmd:
            printable_base = {"x.vel": base_cmd.get("x.vel", 0.0),
                              "y.vel": base_cmd.get("y.vel", 0.0),
                              "theta.vel": base_cmd.get("theta.vel", 0.0)}
            printable_z = {"lift_axis.height_mm": z_cmd.get("lift_axis.height_mm", self._now_height_mm)}
            print({**printable_base, **printable_z})

        return {**base_cmd, **z_cmd}


