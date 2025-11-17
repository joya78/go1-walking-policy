#!/bin/bash
# Afficher un résumé des performances du training

LOG_DIR="/home/maxime/IsaacLab-main/logs/rsl_rl/unitree_go1_rough/2025-11-16_17-17-58"

echo "======================================"
echo "Go1 Training Summary"
echo "======================================"
echo ""

# Compter les checkpoints
CHECKPOINTS=$(ls -1 "$LOG_DIR"/model_*.pt 2>/dev/null | wc -l)
LATEST_MODEL=$(ls -1t "$LOG_DIR"/model_*.pt 2>/dev/null | head -1)

echo "📊 Training Progress:"
echo "  Total checkpoints: $CHECKPOINTS"
echo "  Latest model: $(basename "$LATEST_MODEL")"
echo ""

# Taille des modèles
echo "💾 Model Size:"
if [ -f "$LATEST_MODEL" ]; then
    SIZE=$(du -h "$LATEST_MODEL" | cut -f1)
    echo "  $SIZE"
fi
echo ""

echo "📁 Log Directory:"
echo "  $LOG_DIR"
echo ""

echo "🎯 To visualize:"
echo "  1. TensorBoard: ssh -L 6006:localhost:6006 maxime@server"
echo "     Then open: http://localhost:6006"
echo ""
echo "  2. Download graphs: scp maxime@server:/home/maxime/my_go1_project/videos/training_metrics.png ."
echo ""
echo "  3. Test policy: bash test.sh --checkpoint $LATEST_MODEL"
echo ""
