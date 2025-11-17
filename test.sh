#!/bin/bash
# Quick start script for testing Go1 walking policy

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Go1 Walking Policy Testing${NC}"
echo -e "${BLUE}========================================${NC}"

# Set Isaac Lab directory
ISAACLAB_DIR="/home/ethan/IsaacLab"

# Check if Isaac Lab directory exists
if [ ! -f "$ISAACLAB_DIR/isaaclab.sh" ]; then
    echo -e "${YELLOW}Error: Isaac Lab not found at $ISAACLAB_DIR${NC}"
    exit 1
fi

cd "$ISAACLAB_DIR"

# Activate conda environment
echo -e "\n${GREEN}Activating conda environment...${NC}"
source /home/ethan/miniconda3/etc/profile.d/conda.sh
conda activate isaac_lab

# Default parameters
TASK="Isaac-Velocity-Rough-Unitree-Go1-Play-v0"
NUM_ENVS=50
CHECKPOINT=""
VIDEO=""
VIDEO_LENGTH=500

# Auto-detect latest checkpoint if not specified
LOG_DIR="logs/rsl_rl/unitree_go1_rough"
if [ -z "$CHECKPOINT" ]; then
    if [ -d "$LOG_DIR" ]; then
        LATEST_RUN=$(ls -t "$LOG_DIR" | head -n 1)
        if [ -n "$LATEST_RUN" ]; then
            LATEST_MODEL=$(ls -t "$LOG_DIR/$LATEST_RUN"/model_*.pt | head -n 1)
            if [ -n "$LATEST_MODEL" ]; then
                CHECKPOINT="$LATEST_MODEL"
                echo -e "${GREEN}Auto-detected checkpoint: $CHECKPOINT${NC}"
            fi
        fi
    fi
fi

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
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --video)
            VIDEO="--video"
            shift
            ;;
        --video_length)
            VIDEO_LENGTH="$2"
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
            echo "  --task TASK              Environment to test (default: Isaac-Velocity-Rough-Unitree-Go1-Play-v0)"
            echo "  --num_envs NUM           Number of parallel environments (default: 50)"
            echo "  --checkpoint PATH        Path to model checkpoint (auto-detects latest if not specified)"
            echo "  --video                  Record video of the policy"
            echo "  --video_length LENGTH    Video length in steps (default: 500)"
            echo "  --gui                    Run with GUI (default: headless)"
            echo "  --help                   Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Test with auto-detected latest checkpoint"
            echo "  $0 --checkpoint logs/rsl_rl/unitree_go1_rough/2025-11-15_18-38-41/model_2349.pt"
            echo "  $0 --video --num_envs 10             # Record video with fewer envs"
            echo "  $0 --gui                              # Run with GUI visualization"
            exit 0
            ;;
        *)
            echo -e "${YELLOW}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if checkpoint exists
if [ -z "$CHECKPOINT" ]; then
    echo -e "${YELLOW}Error: No checkpoint found. Please specify with --checkpoint${NC}"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo -e "${YELLOW}Error: Checkpoint not found: $CHECKPOINT${NC}"
    exit 1
fi

# Build the command
CMD="bash isaaclab.sh -p /home/ethan/go1-walking-policy/scripts/play_go1_walking.py \
    --task $TASK \
    --num_envs $NUM_ENVS \
    --checkpoint $CHECKPOINT"

# Add optional parameters
if [ -n "$VIDEO" ]; then
    CMD="$CMD $VIDEO --video_length $VIDEO_LENGTH"
fi

# Display configuration
echo -e "\n${BLUE}Configuration:${NC}"
echo -e "  Task: ${GREEN}$TASK${NC}"
echo -e "  Number of environments: ${GREEN}$NUM_ENVS${NC}"
echo -e "  Checkpoint: ${GREEN}$CHECKPOINT${NC}"
if [ -n "$VIDEO" ]; then
    echo -e "  Video recording: ${GREEN}Enabled (length: $VIDEO_LENGTH steps)${NC}"
else
    echo -e "  Video recording: ${YELLOW}Disabled${NC}"
fi

echo -e "\n${BLUE}Starting policy testing...${NC}"
echo -e "${YELLOW}Command: $CMD${NC}\n"

# Run the command
eval $CMD

echo -e "\n${GREEN}Testing complete!${NC}"
if [ -n "$VIDEO" ]; then
    echo -e "${GREEN}Videos saved to: /home/ethan/go1-walking-policy/videos/${NC}"
fi
