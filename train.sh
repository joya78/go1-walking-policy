#!/bin/bash
# Quick start script for training Go1 walking policy

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Go1 Walking Policy Training${NC}"
echo -e "${BLUE}========================================${NC}"

# Set Isaac Lab directory
ISAACLAB_DIR="/home/maxime/IsaacLab-main"

# Check if Isaac Lab directory exists
if [ ! -f "$ISAACLAB_DIR/isaaclab.sh" ]; then
    echo -e "${YELLOW}Error: Isaac Lab not found at $ISAACLAB_DIR${NC}"
    exit 1
fi

cd "$ISAACLAB_DIR"

# Activate conda environment
echo -e "\n${GREEN}Activating conda environment...${NC}"
source /data/home/maxime/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# Default parameters
TASK="Isaac-Velocity-Rough-Unitree-Go1-v0"
NUM_ENVS=4096
MAX_ITERATIONS=1500
HEADLESS="--headless"
RESUME=""
CHECKPOINT=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK="$2"
            shift 2
            ;;
        --num_envs)
            NUM_ENVS="$2"
            shift 2
            ;;
        --max_iterations)
            MAX_ITERATIONS="$2"
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
        --gui)
            HEADLESS=""
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --task TASK             Environment task (default: Isaac-Velocity-Rough-Unitree-Go1-v0)"
            echo "  --num_envs NUM          Number of environments (default: 4096)"
            echo "  --max_iterations NUM    Training iterations (default: 1500)"
            echo "  --resume                Resume from latest checkpoint"
            echo "  --checkpoint PATH       Resume from specific checkpoint"
            echo "  --gui                   Enable GUI (default: headless)"
            echo "  --help                  Show this help message"
            echo ""
            echo "Available tasks:"
            echo "  Isaac-Velocity-Rough-Unitree-Go1-v0"
            echo "  Isaac-Velocity-Flat-Unitree-Go1-v0"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print configuration
echo -e "\n${GREEN}Training Configuration:${NC}"
echo "  Task: $TASK"
echo "  Number of Environments: $NUM_ENVS"
echo "  Max Iterations: $MAX_ITERATIONS"
echo "  Mode: $([ -z "$HEADLESS" ] && echo "GUI" || echo "Headless")"
[ -n "$RESUME" ] && echo "  Resume: Yes (auto-detect latest)"
[ -n "$CHECKPOINT" ] && echo "  Checkpoint: $CHECKPOINT"

# Create logs directory if it doesn't exist
mkdir -p logs

# Start training
echo -e "\n${GREEN}Starting training...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop training${NC}\n"

bash isaaclab.sh -p /home/maxime/my_go1_project/scripts/train_go1_walking.py \
    --task "$TASK" \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITERATIONS" \
    $RESUME \
    $CHECKPOINT \
    $HEADLESS

echo -e "\n${GREEN}Training completed!${NC}"
echo -e "Logs saved to: ${BLUE}logs/rsl_rl/${NC}"
