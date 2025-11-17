#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
#                    GO1 WALKING POLICY - TRAINING LAUNCHER
# ═══════════════════════════════════════════════════════════════════════════════
#
# This script launches Go1 walking policy training using Isaac Lab.
#
# Prerequisites:
#   - Isaac Lab installed at /home/ethan/IsaacLab
#   - isaac_lab conda environment activated
#   - NVIDIA GPU with CUDA support
#
# Usage:
#   ./launch_training.sh [options]
#
# Options:
#   --num_envs NUM        Number of parallel environments (default: 512)
#   --max_iter NUM        Max training iterations (default: 5000)
#   --headless            Run without GUI (default: enabled)
#   --device DEVICE       GPU device ID (default: 0)
#   --resume              Resume from latest checkpoint
#   --checkpoint PATH     Resume from specific checkpoint
#   --help                Show this help message
#
# Example:
#   ./launch_training.sh --num_envs 2048 --max_iter 10000
#
# ═══════════════════════════════════════════════════════════════════════════════

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No Color

# Default parameters
NUM_ENVS=512
MAX_ITERATIONS=5000
HEADLESS="--headless"
DEVICE=0
RESUME=""
CHECKPOINT=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_envs)
            NUM_ENVS="$2"
            shift 2
            ;;
        --max_iter)
            MAX_ITERATIONS="$2"
            shift 2
            ;;
        --headless)
            HEADLESS="--headless"
            shift
            ;;
        --no-headless)
            HEADLESS=""
            shift
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --resume)
            RESUME="--resume"
            shift
            ;;
        --checkpoint)
            CHECKPOINT="--checkpoint $2"
            shift 2
            ;;
        --help)
            cat "$0" | grep "^#" | tail -n +2
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Print configuration
clear
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}                  GO1 WALKING POLICY - PPO TRAINING LAUNCHER${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${GREEN}Training Configuration:${NC}"
echo -e "  Number of Environments: ${YELLOW}${NUM_ENVS}${NC}"
echo -e "  Training Iterations:    ${YELLOW}${MAX_ITERATIONS}${NC}"
echo -e "  Headless Mode:          ${YELLOW}${HEADLESS//--headless/Enabled}${NC}"
echo -e "  GPU Device:             ${YELLOW}cuda:${DEVICE}${NC}"
if [ -n "$RESUME" ]; then
    echo -e "  Resume Training:        ${YELLOW}Yes${NC}"
fi
echo ""

# Check prerequisites
echo -e "${GREEN}Checking Prerequisites:${NC}"

# Check Isaac Lab directory
if [ ! -d "/home/ethan/IsaacLab" ]; then
    echo -e "${RED}✗ Isaac Lab not found at /home/ethan/IsaacLab${NC}"
    exit 1
else
    echo -e "${GREEN}✓ Isaac Lab found${NC}"
fi

# Check if script exists
SCRIPT_PATH="/home/ethan/go1-walking-policy/scripts/train_go1_walking.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo -e "${RED}✗ Training script not found: $SCRIPT_PATH${NC}"
    exit 1
else
    echo -e "${GREEN}✓ Training script found${NC}"
fi

# Check conda environment
if ! command -v conda &> /dev/null; then
    echo -e "${RED}✗ Conda not found${NC}"
    exit 1
else
    echo -e "${GREEN}✓ Conda available${NC}"
fi

echo ""
echo -e "${GREEN}Checking Isaac Lab Installation:${NC}"

# Navigate to Isaac Lab
cd /home/ethan/IsaacLab

# Verify isaaclab.sh exists
if [ ! -f "isaaclab.sh" ]; then
    echo -e "${RED}✗ isaaclab.sh not found${NC}"
    exit 1
else
    echo -e "${GREEN}✓ isaaclab.sh found${NC}"
fi

# Check Python availability via isaaclab.sh
if ! ./isaaclab.sh -p --version &> /dev/null; then
    echo -e "${YELLOW}⚠ Warning: Could not verify Python version${NC}"
else
    echo -e "${GREEN}✓ Python available via isaaclab.sh${NC}"
fi

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Starting Training...${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════════════${NC}"
echo ""

# Set environment variables
export CUDA_VISIBLE_DEVICES=$DEVICE
export CUDA_LAUNCH_BLOCKING=1

# Run training
echo -e "${YELLOW}Command:${NC}"
echo "  cd /home/ethan/IsaacLab && ./isaaclab.sh -p ../go1-walking-policy/scripts/train_go1_walking.py \\"
echo "    --num_envs $NUM_ENVS --max_iterations $MAX_ITERATIONS $HEADLESS $RESUME $CHECKPOINT"
echo ""

cd /home/ethan/IsaacLab

# Execute training
if ./isaaclab.sh -p ../go1-walking-policy/scripts/train_go1_walking.py \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERATIONS" \
    $HEADLESS \
    $RESUME \
    $CHECKPOINT; then
    
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✓ TRAINING COMPLETED SUCCESSFULLY!${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "Training logs saved to: ${YELLOW}logs/rsl_rl/${NC}"
    echo ""
else
    echo ""
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${RED}✗ TRAINING FAILED${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════════════════════════════${NC}"
    exit 1
fi
