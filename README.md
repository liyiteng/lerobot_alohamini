## Updates
- **[2025-12-11]** Compatible with LeRobot 0.4.3
- **[2025-11-06]** Compatible with LeRobot 0.4


## Introduction

Compared to the original lerobot, lerobot_alohamini significantly enhances debugging capabilities and is adapted for AlohaMini wheeled dual-arm robot hardware (based on Lekiwi extension).

For newly added debugging commands, please refer to:
[Debug Command Summary](examples/debug/README.md)

AlohaMini Hardware 
![alohamini concept](examples/alohamini/media/alohamini3a.png)  


## Getting Started (Ubuntu System)

*** Highly recommended to follow the sequence ***

### 1. Preparation

#### Network Environment Test
```
curl https://www.google.com
curl https://huggingface.co
```
First ensure network connectivity

#### CUDA Environment Test
```
nvidia-smi
```
After entering in terminal, you should be able to see the CUDA version number


### 2. Clone lerobot_alohamini Repository

```
cd ~
git clone https://github.com/liyiteng/lerobot_alohamini.git
```

### 3. Serial Port Authorization
By default, serial ports cannot be accessed. We need to authorize the ports. The lerobot official documentation example modifies serial port permissions to 666, but in practice, this needs to be reset after each computer restart, which is very troublesome. It's recommended to directly add the current user to the device user group for a permanent solution.
1. Enter `whoami` in terminal  // Check current username
2. Enter `sudo usermod -a -G dialout username` // Permanently add username to device user group
3. Restart computer to make permissions effective

### 4. Install conda3 and Environment Dependencies

Install conda3
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

Initialize conda3
```
conda create -y -n lerobot_alohamini python=3.10
conda activate lerobot_alohamini
```

Install environment dependencies
```
cd ~/lerobot_alohamini
pip install -e .[all]
conda install ffmpeg=7.1.1 -c conda-forge
```

### 5. Configure Robot Arm Port Numbers


AlohaMini has 4 robot arms in total: 2 leader arms connected to the PC, 2 follower arms connected to Raspberry Pi, totaling 4 ports.

Since port numbers change every time you reconnect, you must master the operation of finding port numbers. After becoming proficient, you can use hard links for port fixation.

If you purchased the complete AlohaMini machine, the Raspberry Pi that comes with it has already fixed the port numbers for the 2 follower arms, so no additional configuration is needed.


Connect the robot arms to power and to the computer via USB, then find the robot arm port numbers.

Method 1:
Find ports through script:
```
cd ~/lerobot_alohamini

lerobot-find-port
```

Method 2:
You can directly enter commands in terminal and confirm the inserted port numbers by observing the different port numbers displayed after each insertion

```
ls /dev/ttyACM*
```

**After finding the correct ports, please modify the corresponding port numbers in the following files:
Follower arms: lerobot/robots/alohamini/config_lekiwi.py
Leader arms: examples/alohamini/teleoperate_bi.py**

Note: This operation must be performed every time you reconnect the robot arms or restart the computer

### 6. Configure Camera Port Numbers

Camera ports are already built into the Raspberry Pi and do not need configuration:
lerobot/robots/alohamini/config_lekiwi.py


Note:
- Multiple cameras cannot be plugged into one USB Hub; 1 USB Hub only supports 1 camera


### 7. Teleoperation Calibration and Testing


#### 7.1 Set Robot Arm to Middle Position

Host-side calibration:
SSH into the Raspberry Pi, install the conda environment, then perform the following operations:

```
python -m lerobot.robots.alohamini.lekiwi_host
```

If executing for the first time, the system will prompt us to calibrate the robot arm. Position the robot arm as shown in the image, press Enter, then rotate each joint 90 degrees left, then 90 degrees right, then press Enter
![Calibration](examples/alohamini/media/mid_position_so100.png)  


Client-side calibration:
Execute the following command, replace the IP with the actual IP of the host Raspberry Pi, then repeat the above steps
```
python examples/alohamini/teleoperate_bi.py \
--remote_ip 192.168.50.43 \
--leader_id so101_leader_bi

```

#### 7.2 Teleoperation Command Summary

Raspberry Pi side:

```
python -m lerobot.robots.alohamini.lekiwi_host
```

PC side:
```
// Normal teleoperation

python examples/alohamini/teleoperate_bi.py \
--remote_ip 192.168.50.43 \
--leader_id so101_leader_bi


// Teleoperation with voice functionality
python examples/alohamini/teleoperate_bi_voice.py \
--remote_ip 192.168.50.43 \
--leader_id so101_leader_bi


Note: Voice functionality requires installing dependencies and setting DASHSCOPE_API_KEY

// Install voice dependencies
conda install -c conda-forge python-sounddevice
pip install dashscope


// Go to Alibaba Cloud Bailian website, apply for speech recognition API, execute the following command to add the API to environment variables

export DASHSCOPE_API_KEY="sk-434f820ebaxxxxxxxxx"
```

### 8. Record Dataset

#### 1 Register on HuggingFace, Obtain and Configure Key

1. Go to HuggingFace website (huggingface.co), apply for {Key}, remember to include read and write permissions

2. Add API token to Git credentials

```
git config --global credential.helper store

huggingface-cli login --token {key} --add-to-git-credential

```

#### 2 Run Script

Modify the repo-id parameter, then execute:

```
HF_USER=$(huggingface-cli whoami | head -n 1)
echo $HF_USER

```

//Create New Dataset
```
python examples/alohamini/record_bi.py \
  --dataset $HF_USER/so100_bi_test \
  --num_episodes 1 \
  --fps 30 \
  --episode_time 45 \
  --reset_time 8 \
  --task_description "pickup1" \
  --remote_ip 127.0.0.1 \
  --leader_id so101_leader_bi

```

//Resume Dataset
```
python examples/alohamini/record_bi.py \
  --dataset $HF_USER/so100_bi_test \
  --num_episodes 1 \
  --fps 30 \
  --episode_time 45 \
  --reset_time 8 \
  --task_description "pickup1" \
  --remote_ip 127.0.0.1 \
  --leader_id so101_leader_bi \
  --resume 

```

### 9. Replay Dataset
```
python examples/alohamini/replay_bi.py  \
--dataset $HF_USER/so100_bi_test \
--episode 0 \
--remote_ip 127.0.0.1
```

### 10. Dataset Visualization
```
  lerobot-dataset-viz \
  --repo-id $HF_USER/so100_bi_test \
  --episode-index 0
```

### 11. Local Training
// ACT

```
lerobot-train \
  --dataset.repo_id=$HF_USER/so100_bi_test \
  --policy.type=act \
  --output_dir=outputs/train/act_your_dataset1 \
  --job_name=act_your_dataset \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.repo_id=liyitenga/act_policy
```


### 12. Remote Training
Using AutoDL as an example:
Apply for an RTX 4070 GPU, select Python 3.8 (Ubuntu 20.04) CUDA 11.8 or above as container image, and log in via terminal
```
// Enter remote terminal, initialize conda
conda init

// Restart terminal, create environment
conda create -y -n lerobot python=3.10
conda activate lerobot

// Academic acceleration
source /etc/network_turbo

// Get lerobot
git clone https://github.com/liyiteng/lerobot_alohamini.git

// Install necessary files
cd ~/lerobot_alohamini
pip install -e ".[feetech]"
```

Run training command

Finally install FileZilla to retrieve the trained files
```
sudo apt install filezilla -y
```

### 13. Evaluate Training Set

Use FileZilla to copy the trained model to local machine, then run the following command:

```
python examples/alohamini/evaluate_bi.py \
  --num_episodes 3 \
  --fps 20 \
  --episode_time 45 \
  --task_description "Pick and place task" \
  --hf_model_id liyitenga/act_policy \
  --hf_dataset_id liyitenga/eval_dataset \
  --remote_ip 127.0.0.1 \
  --robot_id my_alohamini \
  --hf_model_id ./outputs/train/act_your_dataset1/checkpoints/020000/pretrained_model
  
```