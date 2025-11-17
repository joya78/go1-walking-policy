#!/bin/bash
# Setup virtual display and run Go1 policy with screenshot capture

echo "🖥️  Setting up virtual display for visualization..."

# Check if Xvfb is installed
if ! command -v Xvfb &> /dev/null; then
    echo "❌ Xvfb not installed. Installing..."
    sudo apt-get update && sudo apt-get install -y xvfb x11-utils imagemagick
fi

# Kill any existing Xvfb
pkill Xvfb 2>/dev/null

# Start virtual display
export DISPLAY=:99
Xvfb :99 -screen 0 1920x1080x24 &
XVFB_PID=$!
sleep 2

echo "✅ Virtual display started (PID: $XVFB_PID)"

# Activate conda environment
source /data/home/maxime/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# Go to Isaac Lab
cd /home/maxime/IsaacLab-main

# Create output directory
mkdir -p /home/maxime/my_go1_project/videos/screenshots

echo "🎮 Starting Go1 policy playback..."
echo "📸 Screenshots will be saved to: /home/maxime/my_go1_project/videos/screenshots/"
echo ""

# Run the policy with GUI enabled
python -c "
import os
os.environ['DISPLAY'] = ':99'

import sys
sys.path.insert(0, '/home/maxime/my_go1_project')

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_envs', type=int, default=1)
parser.add_argument('--task', type=str, default='Isaac-Velocity-Rough-Unitree-Go1-Play-v0')
parser.add_argument('--checkpoint', type=str, default='/home/maxime/IsaacLab-main/logs/rsl_rl/unitree_go1_rough/2025-11-16_17-17-58/model_499.pt')

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args([])

# Launch with rendering enabled
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
import time
import subprocess

# Create environment
env_cfg = parse_env_cfg(args.task, device='cuda:0', num_envs=1)
env = gym.make(args.task, cfg=env_cfg)
env = RslRlVecEnvWrapper(env)

# Load policy
agent_cfg_entry_point = gym.spec(args.task).kwargs.get('rsl_rl_cfg_entry_point')
if isinstance(agent_cfg_entry_point, str):
    module_path, class_name = agent_cfg_entry_point.rsplit(':', 1)
    module = __import__(module_path, fromlist=[class_name])
    agent_cfg = getattr(module, class_name)()
else:
    agent_cfg = agent_cfg_entry_point()

runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device='cuda:0')
runner.load(args.checkpoint)

print('[INFO] Policy loaded. Running for 500 steps with screenshots...')

obs, _ = env.get_observations()
screenshot_dir = '/home/maxime/my_go1_project/videos/screenshots'

try:
    for step in range(500):
        with torch.no_grad():
            actions = runner.alg.act(obs, deterministic=True)
        obs, _, _, _ = env.step(actions)
        
        # Take screenshot every 25 steps
        if step % 25 == 0:
            screenshot_file = f'{screenshot_dir}/go1_step_{step:04d}.png'
            subprocess.run(['import', '-window', 'root', screenshot_file], check=False)
            if step % 100 == 0:
                print(f'  📸 Screenshot {step}/500')
    
    print('[INFO] Creating GIF animation...')
    subprocess.run([
        'convert', '-delay', '10', '-loop', '0',
        f'{screenshot_dir}/go1_step_*.png',
        f'{screenshot_dir}/go1_animation.gif'
    ], check=False)
    print(f'✅ Animation saved: {screenshot_dir}/go1_animation.gif')
    
except KeyboardInterrupt:
    print('[INFO] Interrupted by user')

env.close()
simulation_app.close()
" 

# Cleanup
kill $XVFB_PID 2>/dev/null

echo ""
echo "✅ Done! Check your screenshots at:"
echo "   /home/maxime/my_go1_project/videos/screenshots/"
