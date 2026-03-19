#!/bin/bash
# Train the WaveSL LSTM model on the prepared WLASL dataset.
# Run prepare_wlasl.py first if dataset/wlasl does not exist.

set -e

echo "=== WaveSL Model Training ==="
echo ""

if [ ! -d "dataset/wlasl" ]; then
    echo "Error: dataset/wlasl not found."
    echo "Prepare the dataset first:"
    echo "  python src/prepare_wlasl.py --wlasl-dir /path/to/WLASL/start_kit \\"
    echo "      --output-dir dataset/wlasl --class-mapping models/wlasl/class_mapping.json"
    exit 1
fi

if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

mkdir -p models/wlasl dataset/wlasl_cache

echo "Model type: LSTM (seq_len=16)"
echo "Dataset:    dataset/wlasl"
echo "Cache:      dataset/wlasl_cache"
echo "Output:     models/wlasl"
echo ""

python src/train_asl_model.py \
    --data-dir dataset/wlasl \
    --output-dir models/wlasl \
    --cache-dir dataset/wlasl_cache \
    --model-type lstm \
    --seq-len 16 \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --train-split 0.8

echo ""
echo "=== Training Complete ==="
echo "Model:         models/wlasl/best_model.pt"
echo "Class mapping: models/wlasl/class_mapping.json"
echo ""
echo "Run the app:"
echo "  python src/main.py"
