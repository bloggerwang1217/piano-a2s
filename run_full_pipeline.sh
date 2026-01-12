#!/bin/bash
# Full ASAP Test Pipeline
# Usage: ./run_full_pipeline.sh [GPU_ID] [START_STEP]
# Example: ./run_full_pipeline.sh 5 1    (uses GPU 5, start from step 1)
#          ./run_full_pipeline.sh 0 2    (uses GPU 0, start from step 2 - inference)
#          ./run_full_pipeline.sh        (uses GPU 0 by default, start from step 1)

set -e  # Exit on error

# Set GPU (default to 0 if not specified)
GPU_ID=${1:-0}
# Set start step (default to 1 if not specified)
START_STEP=${2:-1}
export CUDA_VISIBLE_DEVICES=$GPU_ID

cd /home/bloggerwang/piano-a2s
source ~/miniconda3/bin/activate a2s2024

# Add humextra and verovio tools to PATH (AFTER conda activate)
export PATH=/home/bloggerwang/piano-a2s/humextra/bin:/home/bloggerwang/piano-a2s/verovio/tools:$PATH
export PYTHONPATH=/home/bloggerwang/piano-a2s:$PYTHONPATH

echo "================================================"
echo "Starting Full ASAP Test Pipeline"
echo "Using GPU: $GPU_ID"
echo "Started at: $(date)"
echo "================================================"

# Step 1: Process ASAP dataset
if [ "$START_STEP" -le 1 ]; then
  echo ""
  echo "[1/3] Processing ASAP dataset..."
  echo "This will generate test data in workspace/feature.asap/test/"
  python datasets/asap.py
  echo "✓ ASAP data processing complete!"
fi

# Step 2: Run inference
if [ "$START_STEP" -le 2 ]; then
  echo ""
  echo "[2/3] Running inference on test set..."
  echo "Loading checkpoint from workspace/1234/finetune.epr/save/"
  python test_inference.py
  echo "✓ Inference complete!"
fi

# Step 3: Generate MusicXML and evaluate
if [ "$START_STEP" -le 3 ]; then
  echo ""
  echo "[3/3] Generating MusicXML files and calculating metrics..."
  python evaluate.py
  echo "✓ Evaluation complete!"
fi

echo ""
echo "================================================"
echo "Pipeline Complete!"
echo "Finished at: $(date)"
echo "================================================"
echo ""
echo "Results location:"
echo "  - Predictions: workspace/1234/finetune.epr/results/test/*.json"
echo "  - MusicXML (pred): workspace/1234/finetune.epr/results/scores/pred/*.xml"
echo "  - MusicXML (target): workspace/1234/finetune.epr/results/scores/target/*.xml"
echo "  - MIDI files: workspace/1234/finetune.epr/results/midi/"
echo "  - MV2H scores: workspace/1234/finetune.epr/results/mv2h/"
echo ""
echo "You can now use the MusicXML files for STEPn evaluation!"
