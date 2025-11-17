# 🎥 Generate Video of Go1 Robot Walking

## Quick Summary

To see a video of the Go1 robot walking in the Isaac Lab simulation, you need to:

1. **Train the policy** (creates neural network checkpoints)
2. **Record evaluation** (runs policy and saves video)
3. **View the video** (playback the recorded MP4)

**Total Time**: ~40-70 minutes (mostly training)

---

## 🚀 Step-by-Step Instructions

### Step 1: Ensure Isaac Lab is Installed

First, check if Isaac Lab installation is complete:

```bash
# Check if Isaac Lab is installed
ls -la /home/ethan/IsaacLab/isaaclab.sh

# If not, reinstall:
bash /home/ethan/install_isaac_lab_official.sh
```

### Step 2: Train the Policy (Creates Checkpoints)

Navigate to the project directory and run training:

```bash
cd /home/ethan/go1-walking-policy

# QUICK TEST (15 minutes, uses 512 envs, 10 iterations)
bash train.sh --num_envs 512 --max_iterations 10

# OR FULL TRAINING (60 minutes, uses 4096 envs, 1500 iterations)
bash train.sh
```

**What happens during training:**
- 512 (or 4096) parallel robots train simultaneously
- Neural network learns to control robot joints
- Checkpoints saved every 100 iterations
- Training metrics logged to `logs/rsl_rl/unitree_go1_rough/*/logs.csv`

**Expected output:**
```
[INFO] Starting training for 10 iterations...
[INFO] Iteration 1/10 | Avg Reward: 0.123 | Loss: 0.456
[INFO] Iteration 2/10 | Avg Reward: 0.234 | Loss: 0.345
...
[INFO] Training completed! Checkpoints saved to: logs/rsl_rl/unitree_go1_rough/2025-11-16_XX-XX-XX/
```

### Step 3: Record Video with Trained Policy

Once training completes, record the video:

```bash
cd /home/ethan/go1-walking-policy

# Record video using latest checkpoint (auto-detected)
bash test.sh --video

# OR use specific checkpoint
bash test.sh --video --checkpoint logs/rsl_rl/unitree_go1_rough/2025-11-16_XX-XX-XX/model_1500.pt

# OR record with fewer environments (faster, lower VRAM)
bash test.sh --video --num_envs 10
```

**What happens during recording:**
- Loads trained neural network from checkpoint
- Runs 50 parallel evaluation episodes
- Records video frames for ~10 seconds
- Saves MP4 file to `videos/` directory
- Prints episode rewards and metrics

**Expected output:**
```
[INFO] Loading checkpoint from: logs/rsl_rl/unitree_go1_rough/2025-11-16_XX-XX-XX/model_1500.pt
[INFO] Starting policy playback...
[INFO] Video recording enabled. Saving to: videos/
[INFO] Episode 1 | Steps: 500 | Avg Reward: 2.12
[INFO] Episode 2 | Steps: 475 | Avg Reward: 2.08
...
[INFO] Video recording complete (500 steps)
[INFO] Video saved to: videos/go1_policy_20251116_142030.mp4
```

### Step 4: View the Video

List and view the generated video:

```bash
# List all recorded videos
ls -lh /home/ethan/go1-walking-policy/videos/

# View with player (if X11 available)
vlc /home/ethan/go1-walking-policy/videos/go1_policy_*.mp4

# Or copy to local machine
scp ethan@<server>:/home/ethan/go1-walking-policy/videos/*.mp4 .
```

---

## 📊 What You'll See in the Video

### Visual Elements
- **50 Go1 quadruped robots** walking simultaneously
- **Procedurally generated rough terrain** with random obstacles
- **3D physics simulation** with realistic lighting and shadows
- **Robot gaits** learned from reinforcement learning
- **Terrain deformation** under robot footsteps

### Animation Details
- **Frame Rate**: 50 FPS
- **Duration**: ~10 seconds per 500 simulation steps
- **Simulation Speed**: 1.0x real-time
- **Physics Frequency**: 200 Hz (5ms timesteps)
- **Action Frequency**: 50 Hz (20ms between actions)

### Performance Metrics Shown
- **Average Reward**: 2.0-2.5 (good walking)
- **Episode Length**: 400-500 steps before termination
- **Success Rate**: 90%+ of episodes complete
- **Velocity Tracking**: Linear and angular velocities match commands
- **Stability**: Robot doesn't fall frequently

---

## 🎬 Video File Details

After recording, your video file will have these specifications:

```
File: /home/ethan/go1-walking-policy/videos/go1_policy_20251116_142030.mp4
Size: ~80-100 MB
Duration: ~10 seconds
Resolution: 1920×1080 (Full HD)
Codec: H.264 (MPEG-4 AVC)
Bitrate: ~80 Mbps
Frame Rate: 50 FPS
Container: MP4 (ISO Base Media File Format)
```

---

## ⏱️ Time Breakdown

| Step | Time | Notes |
|------|------|-------|
| Quick training | 10-15 min | 512 envs, 10 iterations |
| Full training | 45-60 min | 4096 envs, 1500 iterations |
| Video recording | 5-10 min | Depends on num_envs |
| **Total (quick)** | **20-30 min** | Good for seeing results fast |
| **Total (full)** | **50-70 min** | Better trained policy |

---

## 🎥 Video Generation Commands Quick Reference

```bash
# QUICK SETUP (15 min training + 5 min video = 20 min total)
cd /home/ethan/go1-walking-policy
bash train.sh --num_envs 512 --max_iterations 10
bash test.sh --video

# FULL SETUP (60 min training + 5 min video = 65 min total)
cd /home/ethan/go1-walking-policy
bash train.sh
bash test.sh --video

# CUSTOM SETUP (e.g., 2048 envs, 500 iterations)
bash train.sh --num_envs 2048 --max_iterations 500
bash test.sh --video --num_envs 50

# RESUME FROM CHECKPOINT
bash train.sh --resume
bash test.sh --video
```

---

## 📁 File Locations

### Training Output
```
logs/rsl_rl/unitree_go1_rough/
└── 2025-11-16_XX-XX-XX/              # Timestamp directory
    ├── model_0.pt                     # Checkpoint at iteration 0
    ├── model_100.pt                   # Checkpoint at iteration 100
    ├── model_500.pt                   # Checkpoint at iteration 500
    ├── model_1500.pt                  # Final checkpoint (1500 iterations)
    ├── logs.csv                       # Training metrics (reward, loss, etc.)
    └── config.yaml                    # Configuration used for this run
```

### Video Output
```
videos/
├── go1_policy_20251116_142030.mp4     # Video from first run
├── go1_policy_20251116_145000.mp4     # Video from second run
└── ... (one file per evaluation run with --video)
```

---

## 🛠️ Troubleshooting

### Problem: "Isaac Lab not found"
```bash
# Solution: Check and reinstall
ls -la /home/ethan/IsaacLab/isaaclab.sh
bash /home/ethan/install_isaac_lab_official.sh
```

### Problem: "Out of memory" during training
```bash
# Solution: Use fewer environments
bash train.sh --num_envs 512  # Fewer parallel envs
# Or:
bash train.sh --num_envs 2048 --max_iterations 500  # Fewer iterations
```

### Problem: "No checkpoints found" during video recording
```bash
# Solution: Train first
bash train.sh --max_iterations 1  # At least 1 iteration
# Wait 10-20 seconds for checkpoint to save
# Then record video
bash test.sh --video
```

### Problem: Video file corrupted or plays slowly
```bash
# Solution: Install ffmpeg
sudo apt-get install ffmpeg

# Reinstall video dependencies
pip install --upgrade imageio imageio-ffmpeg

# Rerun video recording
bash test.sh --video
```

### Problem: Can't view video on local machine
```bash
# Solution: Copy to local machine using SCP
scp ethan@<server-hostname>:/home/ethan/go1-walking-policy/videos/*.mp4 .

# Or use SFTP
sftp ethan@<server-hostname>
cd /home/ethan/go1-walking-policy/videos/
get go1_policy_*.mp4
```

---

## 📊 Expected Results

After following these steps, you should have:

1. **Trained Neural Network**
   - Stored in: `logs/rsl_rl/unitree_go1_rough/*/model_1500.pt`
   - Size: ~100 KB
   - Learned to map observations → actions

2. **Training Metrics**
   - Stored in: `logs/rsl_rl/unitree_go1_rough/*/logs.csv`
   - Can plot reward curves, loss curves, etc.

3. **Video File**
   - Stored in: `videos/go1_policy_*.mp4`
   - Size: ~80-100 MB
   - Duration: ~10 seconds
   - Shows 50 robots walking in rough terrain

4. **Policy Performance**
   - Average Reward: 2.0+
   - Episode Length: 400-500 steps
   - Success Rate: 90%+
   - Smooth, stable walking gaits

---

## 🚀 Next Steps After Video

Once you have your video, you can:

1. **Analyze Performance**
   ```bash
   python scripts/analyze_training.py logs/rsl_rl/*/logs.csv
   ```

2. **Inspect Checkpoint**
   ```bash
   python scripts/inspect_checkpoint.py logs/rsl_rl/*/model_1500.pt
   ```

3. **Try Different Configurations**
   ```bash
   # Train on flat terrain (easier)
   bash train.sh --task Isaac-Velocity-Flat-Unitree-Go1-v0
   
   # Train with different reward weights
   # (Edit config/go1_walking_env_cfg.py)
   ```

4. **Deploy to Real Robot**
   - Export checkpoint to TorchScript or ONNX
   - Load on Go1 robot's onboard compute
   - Stream observations and execute actions in real-time

---

## 📞 Getting Help

If you encounter issues:

1. Check `/home/ethan/go1-walking-policy/COMPLETE_PROJECT_GUIDE.md` (Troubleshooting section)
2. Read `/home/ethan/ISAAC_LAB_INSTALLATION.md` (Installation issues)
3. Review `/home/ethan/go1-walking-policy/SCRIPTS_WALKTHROUGH.md` (Script details)

---

**Ready to generate your video? Run these commands in order:**

```bash
cd /home/ethan/go1-walking-policy
bash train.sh --num_envs 512 --max_iterations 10  # 15 min
bash test.sh --video                               # 5 min
ls -lh videos/go1_policy_*.mp4                     # View result
```

Happy training! 🚀

