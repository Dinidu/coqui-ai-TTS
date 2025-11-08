#!/bin/bash

# Sinhala TTS Training Script with Automatic TensorBoard
# This script handles venv activation, tensorboard startup, and training

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   Sinhala TTS Training Launcher${NC}"
echo -e "${BLUE}========================================${NC}"

# Parse mode argument from command line
MODE=""
PREV=""
for arg in "$@"; do
    if [[ "$PREV" == "--mode" ]]; then
        MODE="$arg"
        break
    fi
    PREV="$arg"
done

# Display mode information
echo ""
if [[ "$MODE" == "scratch" ]]; then
    echo -e "${YELLOW}🔧 Mode: TRAINING FROM SCRATCH${NC}"
    echo -e "${YELLOW}   Ignoring pretrained model and config${NC}"
elif [[ "$MODE" == "custom" ]]; then
    echo -e "${BLUE}🎨 Mode: CUSTOM CONFIGURATION${NC}"
    echo -e "${BLUE}   Using user-provided paths${NC}"
else
    # Default mode is finetune
    echo -e "${GREEN}🚀 Mode: FINE-TUNING (default)${NC}"
    if [ -f "models/pretrained/config.json" ]; then
        echo -e "${GREEN}   ✓ Config: models/pretrained/config.json${NC}"
    else
        echo -e "${YELLOW}   ⚠ Config not found, will fall back to scratch${NC}"
    fi
    if [ -f "models/pretrained/model_file.pth" ]; then
        echo -e "${GREEN}   ✓ Model: models/pretrained/model_file.pth${NC}"
    else
        echo -e "${YELLOW}   ⚠ Model not found${NC}"
    fi
fi
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}Error: Virtual environment not found!${NC}"
    echo -e "${YELLOW}Please run ./setup.sh first${NC}"
    exit 1
fi

# Activate virtual environment
echo -e "${GREEN}✓ Activating virtual environment...${NC}"
source venv/bin/activate

# Parse the output path from arguments or use default
OUTPUT_PATH=""
CUSTOM_OUTPUT=false
for i in "$@"; do
    if [[ $CUSTOM_OUTPUT == true ]]; then
        OUTPUT_PATH="$i"
        CUSTOM_OUTPUT=false
    fi
    if [[ $i == "--output_path" ]]; then
        CUSTOM_OUTPUT=true
    fi
done

# If no output path specified, create a timestamped one
if [ -z "$OUTPUT_PATH" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_PATH="output/vits_sinhala_${TIMESTAMP}"
fi

# Ensure output directory will be created by the training script
echo -e "${GREEN}✓ Training output will be saved to: ${OUTPUT_PATH}${NC}"

# Function to cleanup background processes on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down...${NC}"
    
    # Kill TensorBoard if it's running
    if [ ! -z "$TENSORBOARD_PID" ] && kill -0 $TENSORBOARD_PID 2>/dev/null; then
        echo -e "${YELLOW}Stopping TensorBoard (PID: $TENSORBOARD_PID)...${NC}"
        kill $TENSORBOARD_PID 2>/dev/null || true
    fi
    
    # Kill any other tensorboard processes started by this script
    pkill -f "tensorboard.*--logdir=$OUTPUT_PATH" 2>/dev/null || true
    
    echo -e "${GREEN}✓ Cleanup complete${NC}"
    exit 0
}

# Set up trap to cleanup on script exit
trap cleanup EXIT INT TERM

# Start training (which will create the output directory)
echo -e "${GREEN}✓ Starting training...${NC}"
echo -e "${BLUE}========================================${NC}"

# Run training in background first to create output directory
python train_sinhala.py --output_path "$OUTPUT_PATH" "$@" &
TRAINING_PID=$!

# Wait a moment for the output directory to be created
echo -e "${YELLOW}Waiting for training to initialize...${NC}"
for i in {1..30}; do
    if [ -d "$OUTPUT_PATH" ]; then
        break
    fi
    sleep 1
done

# Check if training is still running
if ! kill -0 $TRAINING_PID 2>/dev/null; then
    echo -e "${RED}Training failed to start!${NC}"
    wait $TRAINING_PID
    exit 1
fi

# Find the TensorBoard log directory (it's created with a timestamp suffix)
TENSORBOARD_DIR=""
for i in {1..30}; do
    TENSORBOARD_DIR=$(find "$OUTPUT_PATH" -maxdepth 1 -type d -name "vits_sinhala*" 2>/dev/null | head -1)
    if [ ! -z "$TENSORBOARD_DIR" ]; then
        # Wait for tensorboard events file to be created
        for j in {1..10}; do
            if find "$TENSORBOARD_DIR" -name "*.tfevents.*" 2>/dev/null | head -1 | grep -q .; then
                break
            fi
            sleep 1
        done
        break
    fi
    sleep 2
done

# Start TensorBoard if log directory exists
TENSORBOARD_PID=""
if [ ! -z "$TENSORBOARD_DIR" ]; then
    echo -e "${GREEN}✓ Starting TensorBoard...${NC}"
    
    # Find an available port starting from 6006
    PORT=6006
    while lsof -i:$PORT > /dev/null 2>&1; do
        PORT=$((PORT + 1))
    done
    
    # Start TensorBoard in the background
    tensorboard --logdir="$TENSORBOARD_DIR" --port=$PORT --bind_all > /tmp/tensorboard_${TIMESTAMP}.log 2>&1 &
    TENSORBOARD_PID=$!
    
    # Wait for TensorBoard to start
    sleep 3
    
    if kill -0 $TENSORBOARD_PID 2>/dev/null; then
        echo -e "${GREEN}✓ TensorBoard started successfully!${NC}"
        echo -e "${BLUE}========================================${NC}"
        echo -e "${GREEN}  TensorBoard URL: ${BLUE}http://localhost:${PORT}${NC}"
        echo -e "${GREEN}  Logs directory: ${BLUE}${OUTPUT_PATH}${NC}"
        echo -e "${BLUE}========================================${NC}"
    else
        echo -e "${YELLOW}⚠ TensorBoard failed to start (training continues)${NC}"
        TENSORBOARD_PID=""
    fi
else
    echo -e "${YELLOW}⚠ TensorBoard directory not found (training continues)${NC}"
fi

# Bring training back to foreground
echo -e "${GREEN}✓ Training in progress...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop training and TensorBoard${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Wait for training to complete
wait $TRAINING_PID
TRAINING_EXIT_CODE=$?

# Check training exit status
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ Training completed successfully!${NC}"
    echo -e "${GREEN}  Model saved to: ${BLUE}${OUTPUT_PATH}${NC}"
    if [ ! -z "$TENSORBOARD_PID" ] && kill -0 $TENSORBOARD_PID 2>/dev/null; then
        echo -e "${GREEN}  TensorBoard still running at: ${BLUE}http://localhost:${PORT}${NC}"
        echo -e "${YELLOW}  Press Ctrl+C to stop TensorBoard${NC}"
        # Keep script running to maintain TensorBoard
        wait $TENSORBOARD_PID
    fi
    echo -e "${GREEN}========================================${NC}"
else
    echo -e "\n${RED}✗ Training failed with exit code: $TRAINING_EXIT_CODE${NC}"
    exit $TRAINING_EXIT_CODE
fi