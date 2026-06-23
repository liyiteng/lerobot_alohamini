# AlohaMini — Full Workflow

> **Prerequisites:** complete [install.md](install.md) first.  
> **Hardware profiles:** see [profiles.md](profiles.md).

Dual-arm setup — PC (client) + Raspberry Pi (host) on the same LAN.

---

## 1. System Architecture

```
┌──────────────────────────────┐        LAN        ┌──────────────────────────────────┐
│         PC (Client)          │ ◄───────────────► │      Raspberry Pi (Host)         │
│                              │                   │                                  │
│  • Leader arms (USB)         │                   │  • Follower arms (USB)           │
│  • teleoperate_bi.py         │                   │  • Base wheels + lift (USB)      │
│  • record_bi.py              │                   │  • Cameras (USB)                 │
│  • Training / Evaluation     │                   │  • lekiwi_host.py                │
└──────────────────────────────┘                   └──────────────────────────────────┘
```

Both machines must be on the same LAN with the full environment installed.

---

## 2. Port Configuration

Plug in one device at a time, then run:

```bash
lerobot-find-port
# or check directly:
ls /dev/ttyACM*
```

**Follower arms** — edit `src/lerobot/robots/alohamini/config_lekiwi.py` on the Pi:

```python
@dataclass
class LeKiwiConfig(RobotConfig):
    left_port:  str = "/dev/ttyACM0"   # replace with your left-bus port
    right_port: str = "/dev/ttyACM1"   # replace with your right-bus port
```

**Leader arms** — edit `examples/alohamini/teleoperate_bi.py` on the PC:

```python
left_arm_config  = SOLeaderConfig(port="/dev/ttyACM2", ...)   # replace
right_arm_config = SOLeaderConfig(port="/dev/ttyACM3", ...)   # replace
```

> Port numbers can change after reconnecting or rebooting. If you purchased a complete AlohaMini, the Pi's follower ports are already fixed via udev rules — no action needed.

## 3. Camera Configuration

```bash
lerobot-find-cameras
```

Fill the detected index into `src/lerobot/robots/alohamini/config_lekiwi.py`.

> Each camera requires its own USB port — do not share a USB hub between multiple cameras.

---

## 4. Calibration

### Step 1 — Calibrate follower arms (Pi side)

SSH into the Pi and run the host script for your model. On first run, the script prompts calibration: position each joint at its mechanical midpoint → Enter → rotate 90° left → Enter → rotate 90° right → Enter.

```bash
# AlohaMini 1 (SO-ARM 5-DoF)
python -m lerobot.robots.alohamini.lekiwi_host --robot_model alohamini1

# AlohaMini 2 (AM-ARM 6-DoF)
python -m lerobot.robots.alohamini.lekiwi_host --robot_model alohamini2

# AlohaMini 2 Pro (AM-ARM 6-DoF, STS3250)
python -m lerobot.robots.alohamini.lekiwi_host --robot_model alohamini2pro
```

SO-ARM 5-DoF reference middle position:

![Calibration SO-ARM](../../examples/alohamini/media/mid_position_so100.png)

### Step 2 — Calibrate leader arms (PC side)

Replace `<Pi_IP>` with your Raspberry Pi's IP address.

SO-ARM leader (5-DoF):

```bash
python examples/alohamini/teleoperate_bi.py \
  --remote_ip <Pi_IP> \
  --robot_model alohamini1 \
  --leader_id so101_leader_bi \
  --arm_profile so-arm-5dof
```

AM-ARM leader (6-DoF):

```bash
python examples/alohamini/teleoperate_bi.py \
  --remote_ip <Pi_IP> \
  --robot_model alohamini2 \
  --leader_id am_leader_bi \
  --arm_profile am-leader-6dof
```

> Power-cycle both leader and follower arms after calibration for changes to take effect.

---

## 5. Teleoperation

Start the Pi host first, then the PC client (calibration is skipped since it's already done):

```bash
# Pi — run the host for your robot:
python -m lerobot.robots.alohamini.lekiwi_host --robot_model alohamini1
python -m lerobot.robots.alohamini.lekiwi_host --robot_model alohamini2
python -m lerobot.robots.alohamini.lekiwi_host --robot_model alohamini2pro

# PC — run the client for your leader arm:
python examples/alohamini/teleoperate_bi.py \
  --remote_ip <Pi_IP> --robot_model alohamini1 --leader_id so101_leader_bi --arm_profile so-arm-5dof

python examples/alohamini/teleoperate_bi.py \
  --remote_ip <Pi_IP> --robot_model alohamini2 --leader_id am_leader_bi --arm_profile am-leader-6dof
```

---

## 6. Dataset Recording

> Make sure the Pi host is already running (§5) before recording.  
> `--arm_profile` here refers to your **leader arm** hardware, not the follower robot.  
> `--robot_model` must match the model running on the Pi host.  
> Replace `<Pi_IP>` with your Raspberry Pi's IP address.
> `record_bi.py` prints the local dataset path and uploads to Hugging Face Hub by default. Add `--push_to_hub false` to keep the dataset local only.

### AlohaMini 1 — SO-ARM leader (5-DoF)

Create new dataset:

```bash
python examples/alohamini/record_bi.py \
  --dataset $HF_USER/so100_bi_test \
  --num_episodes 1 \
  --fps 30 \
  --episode_time 45 \
  --reset_time 8 \
  --task_description "pickup1" \
  --remote_ip <Pi_IP> \
  --robot_model alohamini1 \
  --leader_id so101_leader_bi \
  --arm_profile so-arm-5dof
```

Resume existing dataset (add `--resume`):

```bash
python examples/alohamini/record_bi.py \
  --dataset $HF_USER/so100_bi_test \
  --num_episodes 1 \
  --fps 30 \
  --episode_time 45 \
  --reset_time 8 \
  --task_description "pickup1" \
  --remote_ip <Pi_IP> \
  --robot_model alohamini1 \
  --leader_id so101_leader_bi \
  --arm_profile so-arm-5dof \
  --resume
```

### AlohaMini 2 / 2 Pro — AM-ARM leader (6-DoF)

Create new dataset:

```bash
python examples/alohamini/record_bi.py \
  --dataset $HF_USER/am2_bi_test \
  --num_episodes 1 \
  --fps 30 \
  --episode_time 45 \
  --reset_time 8 \
  --task_description "pickup1" \
  --remote_ip <Pi_IP> \
  --robot_model alohamini2 \
  --leader_id am_leader_bi \
  --arm_profile am-leader-6dof
```

Resume existing dataset (add `--resume`):

```bash
python examples/alohamini/record_bi.py \
  --dataset $HF_USER/am2_bi_test \
  --num_episodes 1 \
  --fps 30 \
  --episode_time 45 \
  --reset_time 8 \
  --task_description "pickup1" \
  --remote_ip <Pi_IP> \
  --robot_model alohamini2 \
  --leader_id am_leader_bi \
  --arm_profile am-leader-6dof \
  --resume
```

---

## 7. Dataset Replay

```bash
python examples/alohamini/replay_bi.py \
  --dataset $HF_USER/am2_bi_test \
  --episode 0 \
  --remote_ip <Pi_IP> \
  --robot_model alohamini2
```

---

## 8. Dataset Visualization

```bash
lerobot-dataset-viz \
  --repo-id $HF_USER/am2_bi_test \
  --episode-index 0 \
  --display-compressed-images
```

---

## 9. Training

### Local training

```bash
lerobot-train \
  --dataset.repo_id=$HF_USER/am2_bi_test \
  --policy.type=act \
  --output_dir=outputs/train/act_your_dataset1 \
  --job_name=act_your_dataset \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.repo_id=$HF_USER/act_policy \
  --dataset.video_backend=pyav
```

### No local GPU?

Use any cloud GPU provider (e.g. AutoDL, Lambda Labs, Vast.ai). Set up the environment the same way as local, run the same training command, then copy the checkpoint back to your machine for evaluation.

---

## 10. Evaluation

Make sure the Pi host is already running (§5), then run inference from the PC.

> `--robot_model` / `--robot.robot_model` must match the model running on the Pi host:  
> `alohamini1` (SO-ARM 5-DoF, 16-dim state) · `alohamini2` / `alohamini2pro` (AM-ARM 6-DoF, 18-dim state)

### `evaluate_bi.py` (custom script, N episodes, records to Hub)

```bash
python examples/alohamini/evaluate_bi.py \
  --num_episodes 3 \
  --fps 20 \
  --episode_time 45 \
  --task_description "Pick and place task" \
  --hf_model_id outputs/train/act_your_dataset1/checkpoints/020000/pretrained_model \
  --hf_dataset_id $HF_USER/eval_act_policy \
  --remote_ip <Pi_IP> \
  --robot_id my_alohamini \
  --robot_model alohamini2
```

---

## 11. Debug

See [Debug Command Summary](../../examples/debug/README.md) for the full list of debugging utilities.
