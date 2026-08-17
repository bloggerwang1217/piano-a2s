#!/usr/bin/env python3
"""
Pretrain Test-Only Inference Script

Loads pretrained checkpoint and runs inference on MuseSyn+HumSyn test set.
Skips training entirely - only needs test split features.

Usage:
    python pretrain_inference.py hparams/pretrain.yaml --device cuda
"""

import sys
import os
import logging
import numpy as np
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from datasets.syn import TestDataset
from pretrain import ASR, calculate_wer, caculate_f1
from utilities import save, load

logger = logging.getLogger(__name__)


class InferenceASR(ASR):
    """ASR subclass that skips optimizer/checkpoint updates in on_stage_end."""

    def on_stage_end(self, stage, stage_loss, epoch):
        stage_stats = {"loss": stage_loss,
                       "time_loss": np.mean(self.time_losses),
                       "key_loss": np.mean(self.key_losses),
                       "upper_loss": np.mean(self.upper_losses),
                       "lower_loss": np.mean(self.lower_losses),
                       "teacher_forcing_ratio": self.teacher_forcing_ratio}

        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
            return

        if not hasattr(self, 'train_stats'):
            self.train_stats = {"loss": -1}

        wer_upper, wer_upper_dict = calculate_wer(self.upper_pred, self.upper_target)
        wer_lower, wer_lower_dict = calculate_wer(self.lower_pred, self.lower_target)
        key_f1, key_f1_dict = caculate_f1(self.key_pred, self.key_target)
        time_f1, time_f1_dict = caculate_f1(self.time_sig_pred, self.time_sig_target)
        stage_stats["key_f1"] = key_f1
        stage_stats["time_f1"] = time_f1
        stage_stats["WER_upper"] = wer_upper
        stage_stats["WER_lower"] = wer_lower
        stage_stats["WER"] = (wer_upper + wer_lower) / 2

        # Log stats without lr_annealing or checkpoint saving
        logger.info(f"Test WER: {stage_stats['WER']:.4f} "
                    f"(upper: {wer_upper:.4f}, lower: {wer_lower:.4f})")
        logger.info(f"Key F1: {key_f1:.4f}, Time F1: {time_f1:.4f}")

        # Save predictions (same as original)
        for id in self.upper_pred:
            pred = []
            for i in range(len(self.upper_pred[id])):
                key_pred = self.key_pred[id][i] - 6
                time_sig_pred = self.time_sig_list[self.time_sig_pred[id][i]]
                lower_pred = self.lower_pred[id][i]
                upper_pred = self.upper_pred[id][i]
                pred.append([key_pred, time_sig_pred, lower_pred, upper_pred])
            version, chunk_name, soundfont = id.split('~')
            split = 'valid' if stage == sb.Stage.VALID else 'test'
            style = 'classical' if chunk_name[0].islower() else 'pop'
            info_path = os.path.join(self.hparams.feature_folder, split, version, 'info', f'{chunk_name}.json')
            info = load(info_path)
            composer = info.get('composer', info['score_name'].split('#')[0])
            wer_upper_val = wer_upper_dict[id]
            wer_lower_val = wer_lower_dict[id]
            key_f1_val = key_f1_dict[id]
            time_f1_val = time_f1_dict[id]
            target_path = os.path.join(self.hparams.feature_folder, split, version, 'target', f'{chunk_name}.pkl')
            result_path = os.path.join(self.hparams.output_folder, 'results', split, f'{id}.json')
            result = {'style': style, 'soundfont': soundfont,
                      'composer': composer, 'target_path': target_path,
                      'pred': pred, 'wer_upper': wer_upper_val, 'wer_lower': wer_lower_val,
                      'key_f1': key_f1_val, 'time_f1': time_f1_val}
            save(result, result_path)


if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Only create test dataset
    test_versions = range(4) if hparams["midi_syn"] == 'epr' else [0]
    test_dataset = TestDataset(hparams, 'test', run_opts["device"], test_versions)
    logger.info(f"Test set size: {len(test_dataset)} samples")

    # Create ASR brain and load checkpoint
    asr_brain = InferenceASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Run evaluation only
    asr_brain.evaluate(
        test_dataset,
        test_loader_kwargs=hparams["test_dataloader_opts"],
        min_key="WER",
    )

    print(f"\nInference complete! Results saved to {hparams['output_folder']}/results/test/")
    print(f"Next: python evaluate.py")
