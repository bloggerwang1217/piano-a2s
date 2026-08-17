#!/usr/bin/env python3
"""
ASAP Test Set Inference Script

Usage:
    python test_inference.py

Description:
    1. Load fine-tuned checkpoint from workspace/1234/finetune.epr/save/
    2. Run inference on ASAP test set
    3. Save predictions to <output_folder>/results/test/*.json
    4. Calculate evaluation metrics (WER, etc.)
"""

import torch
import argparse
from pathlib import Path
from utilities import load
from datasets.asap import ASAPDataset
from models import ScoreTranscription
from finetune import ASR
import sys
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hparams', default='hparams/finetune.yaml')
    ap.add_argument('--checkpoint-dir', default=None,
                    help='Override save_folder (where CKPT+... lives)')
    ap.add_argument('--feature-folder', default=None,
                    help='Override feature_folder (ASAP test data location)')
    ap.add_argument('--output-folder', default=None,
                    help='Override output_folder (where results/test/<id>.json goes)')
    args = ap.parse_args()

    print(f"Loading hyperparameters from {args.hparams}...")
    hparams = load(args.hparams)
    if args.checkpoint_dir:
        ckpt_abs = Path(args.checkpoint_dir).resolve()
        hparams['checkpointer'].checkpoints_dir = ckpt_abs
        hparams['save_folder'] = str(ckpt_abs)
    if args.feature_folder:
        hparams['feature_folder'] = os.path.abspath(args.feature_folder)
    if args.output_folder:
        hparams['output_folder'] = os.path.abspath(args.output_folder)
        os.makedirs(os.path.join(hparams['output_folder'], 'results', 'test'),
                    exist_ok=True)

    # Set device (speechbrain expects string, not torch.device)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create ASR brain
    print("Creating ASR brain...")
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts={"device": device},
        checkpointer=hparams["checkpointer"],
    )

    # Load checkpoint
    print(f"Loading checkpoint from {hparams['save_folder']}...")
    asr_brain.checkpointer.recover_if_possible()

    # Create test dataset
    print("Creating test dataset...")
    test_dataset = ASAPDataset(hparams, "test", device)
    print(f"Test set size: {len(test_dataset)} samples")

    # Run evaluation
    print("Starting evaluation...")
    asr_brain.evaluate(
        test_dataset,
        min_key="WER",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )

    print(f"\nEvaluation complete! Results saved to {hparams['output_folder']}/results/test/")
    print(f"\nNext steps:")
    print(f"1. Run: python evaluate.py")
    print(f"   This will generate MusicXML files in {hparams['output_folder']}/results/scores/")
    print(f"2. Use the generated MusicXML files for STEPn evaluation")


if __name__ == "__main__":
    main()
