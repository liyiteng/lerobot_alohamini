# lift_axis.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Protocol
import time

# ---- 极薄的总线协议（与你现有 Feetech Bus 兼容）----
class BusLike(Protocol):
    motors: Dict[str, object]
    def read(self, item: str, name: str) -> float: ...
    def write(self, item: str, name: str, value: float) -> None: ...
    def sync_write(self, item: str, values: Dict[str, float]) -> None: ...

# ---- 兼容你的电机类型枚举（按项目结构替换 import 路径）----
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import OperatingMode


# ---- 配置（并入左/右现有 bus；不新建 bus）----
@dataclass
class LiftAxisConfig:
    enabled: bool = True
    name: str = "lift_axis"
    bus: str = "left"                 # "left" or "right"（决定用哪条现有 bus）
    motor_id: int = 11
    motor_model: str = "sts3215"

    # 机械换算（1圈=360°=4096tick），按你的丝杆/传动比改
    lead_mm_per_rev: float = 84      # 丝杆导程（每圈上升的毫米）
    output_gear_ratio: float = 1.0    # 舵机角→丝杆角传动比
    soft_min_mm: float = 0.0
    soft_max_mm: float = 600        # 升降行程

    # 归零（下探硬限位→回弹）
    home_down_speed: int = 1000   # 速度模式下向下的目标速度（单位依电机）
    home_stall_current_ma: int = 60 # 堵转电流阈值；没电流反馈时用滞停判据
    home_backoff_deg: float = 5.0

    # 速度闭环
    kp_vel: float = 300               # (目标速度单位 / mm)
    v_max: int = 1000              # 速度上限（单位依电机）
    on_target_mm: int = 1.0         # 到位阈值（mm）
    
    dir_sign: int = -1  # ← 新增，+1 不翻转；-1 翻转方向
    step_mm: float = 10  # 每次按键的步进（mm）



class LiftAxis:
    """并入左/右现有 bus 的 Z 轴控制器（速度模式 + 多圈计数 + 毫米高度闭环）"""
    def __init__(
        self,
        cfg: LiftAxisConfig,
        bus_left: Optional[BusLike],
        bus_right: Optional[BusLike],
    ):
        self.cfg = cfg
        self._bus = (bus_left if cfg.bus == "left" else bus_right)
        self.enabled = bool(cfg.enabled and self._bus is not None)
        self._ticks_per_rev = 4096.0
        self._deg_per_tick = 360.0 / self._ticks_per_rev
        self._mm_per_deg = (cfg.lead_mm_per_rev * cfg.output_gear_ratio) / 360.0

        # 多圈位置（扩展tick）
        self._last_tick: float = 0.0
        self._extended_ticks: float = 0.0  # 连续累计
        # 零位（扩展角度下的度）
        self._z0_deg: float = 0.0

        # 目标（非阻塞）
        self._target_mm: float = 0.0

        self._configured = False

    # ---------- 生命周期：注册/配置 ----------
    def attach(self) -> None:
        if not self.enabled: return
        if self.cfg.name not in self._bus.motors:
            self._bus.motors[self.cfg.name] = Motor(self.cfg.motor_id, self.cfg.motor_model, MotorNormMode.DEGREES)

    def configure(self) -> None:
        if not self.enabled: return

        if self._configured: return   # 已经配过就别再动return
        # 持续旋转 → 速度模式
        self._bus.write("Operating_Mode", self.cfg.name, OperatingMode.VELOCITY.value)
        # 读初值以初始化扩展计数
        self._last_tick = float(self._bus.read("Present_Position", self.cfg.name, normalize=False))
        self._extended_ticks = 0.0
        self._configured = True

    # ---------- 内部：扩展 tick 累计（允许无限圈） ----------
    def _update_extended_ticks(self) -> None:
        if not self.enabled: return
        cur = float(self._bus.read("Present_Position", self.cfg.name, normalize=False))  # 0..4095
        delta = cur - self._last_tick
        half = self._ticks_per_rev * 0.5
        if   delta > +half: delta -= self._ticks_per_rev
        elif delta < -half: delta += self._ticks_per_rev
        self._extended_ticks += delta
        self._last_tick = cur

    # ---------- 单位换算 ----------
    def _extended_deg(self) -> float:
        return self.cfg.dir_sign * self._extended_ticks * self._deg_per_tick 

    def get_height_mm(self) -> float:
        if not self.enabled: return 0.0
        self._update_extended_ticks()
        raw_mm = (self._extended_deg() - self._z0_deg) * self._mm_per_deg
        #print(f"[lift_axis.get_height_mm] raw_mm={raw_mm:.2f}, extended_deg={self._extended_deg():.2f}, z0_deg={self._z0_deg:.2f}")  # debug
        return raw_mm
    
    # ---------- 归零（下探硬限位→回弹，建立 z=0mm） ----------
    def home(self, use_current: bool = True) -> None:
        if not self.enabled: return
        self.configure()
        name = self.cfg.name
        # 向下
        v_down = self.cfg.home_down_speed 
        self._bus.write("Goal_Velocity", name, v_down)
        stuck = 0
        last_tick = int(self._bus.read("Present_Position", name, normalize=False))
        for _ in range(600):  # ~30s @50ms
            time.sleep(0.05)
            self._update_extended_ticks()
            now_tick = self._last_tick
            moved = abs(now_tick - last_tick) > 10
            last_tick = now_tick
            cur_ma = 0
            raw_cur_ma = 0
            if use_current:
                try: 
                    raw_cur_ma = int(self._bus.read("Present_Current", name, normalize=False))
                    cur_ma = raw_cur_ma * 6.5
                    print(f"[lift_axis.home] Present_Current={cur_ma} mA")  # debug
                    print(f"[lift_axis.home] Present_Position={now_tick} ticks")  # debug

                except Exception: cur_ma = 0
            if (use_current and cur_ma >= self.cfg.home_stall_current_ma) or (not moved):
                print(f"[lift_axis.home] Stalled at current={cur_ma} mA, moved={moved}")  # debug
                stuck += 1
            else:
                stuck = 0
            if stuck >= 2: break
        # 停止并回弹一点
        #self._bus.write("Goal_Velocity", name, 0)
        self._bus.write("Torque_Enable", name, 0)
        print("Disable torque output (motor will be released)")
        time.sleep(1)

        self._update_extended_ticks()
        self._z0_deg = self._extended_deg()       
        print("Extended ticks after homing:", self._extended_ticks)
        h_now = self.get_height_mm()
        print(f"[home] set-zero z0_deg={self._z0_deg:.2f}, height_now={h_now:.2f} mm")  # 这里应≈0



    # ---------- 非阻塞：设置目标 + 每帧 update(dt) ----------
    def set_height_target_mm(self, height_mm: float) -> None:
        if not self.enabled: return
        self._target_mm = max(self.cfg.soft_min_mm, min(self.cfg.soft_max_mm, height_mm))

    def clear_target(self) -> None:
        if not self.enabled: return
        self._target_mm = None
        self._bus.write("Goal_Velocity", self.cfg.name, 0.0)

    def update(self) -> None:
        """每帧调用一次（建议在你的主循环里 50~100Hz 调用）"""
        if not self.enabled or self._target_mm is None: return
        cur_mm = self.get_height_mm()
        err = self._target_mm - cur_mm
        # 到位判据
        if abs(err) <= self.cfg.on_target_mm:
            self._bus.write("Goal_Velocity", self.cfg.name, 0)
            self._target_mm = None
            return
        # 简单 P 控制
        v = self.cfg.kp_vel * err
        v = max(-self.cfg.v_max, min(self.cfg.v_max, v))
        self._bus.write("Goal_Velocity", self.cfg.name, int(self.cfg.dir_sign * v))

        # 读电流仅供调试
        raw_cur_ma = int(self._bus.read("Present_Current", self.cfg.name, normalize=False))
        cur_ma = raw_cur_ma * 6.5
        print(f"[lift_axis.update] target={self._target_mm:.2f} mm, cur={cur_mm:.2f} mm, err={err:.2f} mm, v={v:.1f}| current={cur_ma} mA")

    # ---------- 与现有 action/obs 的薄耦合 ----------
    def contribute_observation(self, obs: Dict[str, float]) -> None:
        """导出便于上位机使用的键：height_mm 和 当前速度读数"""
        if not self.enabled: return
        obs[f"{self.cfg.name}.height_mm"] = self.get_height_mm()
        try:
            obs[f"{self.cfg.name}.vel"] = int(self._bus.read("Present_Velocity", self.cfg.name, normalize=False))
        except Exception:
            pass

    def apply_action(self, action: Dict[str, float]) -> None:
        """
        支持两种键：
        - f"{name}.height_mm": 目标高度（mm）（推荐）
        - f"{name}.vel"      : 直接给目标速度（高级用）
        """
        #print(f"[lift_axis.apply_action] action={action}")  # debug
        if not self.enabled: return
        key_h = f"{self.cfg.name}.height_mm"
        key_v = f"{self.cfg.name}.vel"
        if key_h in action:
            self.set_height_target_mm(float(action[key_h]))
        if key_v in action:
            # 直接速度控制会清掉高度目标
            self._target_mm = None
            v = int(action[key_v])
            v = max(-self.cfg.v_max, min(self.cfg.v_max, v))
            # 越界保护：已到上/下限时阻止继续向外运动
            try:
                cur_mm = self.get_height_mm()
                if (cur_mm >= self.cfg.soft_max_mm and v > 0) or (cur_mm <= self.cfg.soft_min_mm and v < 0):
                    v = 0
            except Exception:
                pass
            self._bus.write("Goal_Velocity", self.cfg.name, v * self.cfg.dir_sign)
        
        ticks = int(self._bus.read("Present_Position", self.cfg.name, normalize=False))
        # print(f"[lift_axis] Z-axis ticks: {ticks}")
        # print(f"[lift_axis] Z-axis height: {self.get_height_mm():.2f} mm")
