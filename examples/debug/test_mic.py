# mic_diag.py
import os, sys, time, json, math, tempfile, wave
import numpy as np

def list_devices():
    import sounddevice as sd
    devs = sd.query_devices()
    print("\n=== 可用音频设备（输入设备） ===")
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) > 0:
            print(f"[{i}] {d['name']}  (in={d['max_input_channels']}, out={d.get('max_output_channels',0)}, default_sr={d['default_samplerate']})")
    print("（若没有任何行，说明系统没有可用麦克风或驱动未就绪）\n")
    return devs

def pick_input_device(devs):
    env = os.getenv("VOICE_DEVICE_INDEX")
    if env is not None and env.isdigit():
        idx = int(env)
        if 0 <= idx < len(devs) and devs[idx].get("max_input_channels", 0) > 0:
            return idx
    # 默认优先系统默认输入设备，否则选第一个有输入通道的
    try:
        import sounddevice as sd
        default_in = sd.default.device[0]
        if default_in is not None and devs[default_in].get("max_input_channels", 0) > 0:
            return default_in
    except Exception:
        pass
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) > 0:
            return i
    return None

def record_3s_wav(device_idx, samplerate=16000, channels=1):
    import sounddevice as sd
    sd.default.device = (device_idx, None)
    try:
        sr_dev = int(sd.query_devices(device_idx)["default_samplerate"])
        if abs(sr_dev - samplerate) > 1:
            print(f"⚠️ 设备默认采样率 {sr_dev}Hz，与期望 {samplerate}Hz 不同，先用设备默认 {sr_dev}Hz。")
            samplerate = sr_dev
    except Exception:
        pass
    print(f"[diag] 开始录音… 设备={device_idx}, 采样率={samplerate}Hz, 通道={channels}")
    audio = sd.rec(int(3 * samplerate), samplerate=samplerate, channels=channels, dtype="int16")
    sd.wait()
    # 计算 RMS
    rms = float(np.sqrt(np.mean((audio.astype(np.float32) / 32768.0) ** 2)) + 1e-12)
    dbfs = 20 * math.log10(rms)
    print(f"[diag] 3秒录音完成。RMS={rms:.6f}, 约 {dbfs:.1f} dBFS（<-60 dBFS 往往表示太小/静音/麦克风静音）")
    # 写 wav 文件
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name; tmp.close()
    with wave.open(tmp_path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio.tobytes())
    print(f"[diag] 已保存测试音频：{tmp_path}")
    return tmp_path

def try_asr(wav_path):
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        print(f"⚠️ faster-whisper 未安装或不可用：{e}\n请先执行：pip install faster-whisper")
        return
    print("[diag] 加载 ASR 模型 small（首次会下载，较慢）…")
    try:
        model = WhisperModel("small", device="cpu")
    except Exception as e:
        print("❌ ASR 模型加载失败：", e)
        return
    print("[diag] 开始识别…")
    try:
        segments, _ = model.transcribe(wav_path, language="zh")
        text = "".join(seg.text for seg in segments).strip()
        print(f"[diag] 识别结果：{text!r}" if text else "[diag] 没识别到文本（可能太小/静音/非中文）。")
    except Exception as e:
        print("❌ 识别失败：", e)

def main():
    print(">>> 诊断步骤：1) 枚举设备 2) 录3秒并测量音量 3) 试ASR")
    devs = list_devices()
    if not devs:
        print("❌ sd.query_devices() 无结果。请检查声卡/驱动/容器权限（Linux 可试：sudo apt install alsa-utils pulseaudio; 物理机确认麦克风未静音）")
        return
    idx = pick_input_device(devs)
    if idx is None:
        print("❌ 没找到可用输入设备。可设置环境变量 VOICE_DEVICE_INDEX=设备编号 再试。")
        return
    wav = record_3s_wav(idx)
    try_asr(wav)

if __name__ == "__main__":
    main()
