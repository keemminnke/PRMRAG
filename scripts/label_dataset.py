#!/usr/bin/env python3
"""
Main script for consensus-based auto-labeling pipeline.

This script:
1. Loads RAG-CoT trajectories
2. Labels them with RPE labeler (small model)
3. Labels them with Judge labeler (large model)
4. Applies consensus filtering to produce auto-labeled dataset
5. Saves high-quality training samples
"""

import argparse
import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prmrag.config import load_pipeline_config
from prmrag.data import load_trajectories, save_labeled_trajectories, save_training_samples
from prmrag.labeling import RPELabeler, JudgeLabeler, ConsensusModule
from prmrag.labeling.consensus import print_consensus_report
from prmrag.utils import setup_logger


def main():
    parser = argparse.ArgumentParser(
        description="Consensus-based auto-labeling pipeline for RAG-CoT"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input trajectories (JSONL format)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to output labeled dataset",
    )
    parser.add_argument(
        "--output-training-samples",
        type=Path,
        help="Path to save training samples (optional)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of trajectories to process (for testing)",
    )
    parser.add_argument(
        "--skip-rpe",
        action="store_true",
        help="Skip RPE labeling (load from cache)",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip Judge labeling (load from cache)",
    )
    parser.add_argument(
        "--rpe-cache",
        type=Path,
        help="Path to cached RPE labels",
    )
    parser.add_argument(
        "--judge-cache",
        type=Path,
        help="Path to cached Judge labels",
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_pipeline_config(args.config)

    # Setup logging
    log_file = Path(config.logging.log_dir) / "label_dataset.log" if config.logging.log_to_file else None
    logger = setup_logger("prmrag", level=config.logging.level, log_file=log_file)

    logger.info("=" * 60)
    logger.info("CONSENSUS-BASED AUTO-LABELING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")

    # Load trajectories
    logger.info(f"\n[1/4] Loading trajectories from {args.input}")
    trajectories = load_trajectories(
        args.input,
        format=config.data.input_format,
        limit=args.limit,
    )
    logger.info(f"Loaded {len(trajectories)} trajectories")

    # RPE Labeling
    if not args.skip_rpe:
        logger.info("\n[2/4] Running RPE labeling (MC-based, small model)")
        logger.info(f"  Model: {config.rpe.model_name}")
        logger.info(f"  Rollouts: {config.rpe.num_rollouts}")
        logger.info(f"  Thresholds: GOOD={config.rpe.threshold_good}, BAD={config.rpe.threshold_bad}")

        rpe_labeler = RPELabeler(config.rpe.__dict__)
        rpe_labels_batch = rpe_labeler.label_batch(trajectories, show_progress=True)

        # Optionally save RPE labels
        if config.output.save_individual_labels:
            rpe_output = args.output.parent / f"{args.output.stem}_rpe.jsonl"
            logger.info(f"  Saving RPE labels to {rpe_output}")
            # TODO: Implement save function for just labels

    else:
        logger.info("\n[2/4] Loading cached RPE labels")
        # TODO: Load from cache
        raise NotImplementedError("RPE label caching not yet implemented")

    # Judge Labeling
    if not args.skip_judge:
        logger.info("\n[3/4] Running Judge labeling (LLM-based, large model)")
        logger.info(f"  Model: {config.judge.model_name}")
        logger.info(f"  Prompt style: {config.judge.prompt_style}")

        judge_labeler = JudgeLabeler(config.judge.__dict__)
        judge_labels_batch = judge_labeler.label_batch(trajectories, show_progress=True)

        # Optionally save Judge labels
        if config.output.save_individual_labels:
            judge_output = args.output.parent / f"{args.output.stem}_judge.jsonl"
            logger.info(f"  Saving Judge labels to {judge_output}")
            # TODO: Implement save function for just labels

    else:
        logger.info("\n[3/4] Loading cached Judge labels")
        # TODO: Load from cache
        raise NotImplementedError("Judge label caching not yet implemented")

    # Consensus Filtering
    logger.info("\n[4/4] Applying consensus filtering")
    logger.info(f"  Strategy: {config.consensus.strategy}")
    logger.info(f"  Trajectory-level filtering: {config.consensus.trajectory_level}")
    logger.info(f"  Min agreement: {config.consensus.min_agreement}")

    consensus_module = ConsensusModule(config.consensus.__dict__)

    labeled_trajectories = consensus_module.process_batch(
        trajectories,
        rpe_labels_batch,
        judge_labels_batch,
    )

    # Compute statistics
    stats = consensus_module.compute_statistics(labeled_trajectories)
    print_consensus_report(stats)

    # Save results
    logger.info(f"\nSaving labeled trajectories to {args.output}")
    save_labeled_trajectories(
        labeled_trajectories,
        args.output,
        format=config.output.format,
        include_filtered=False,  # Only save non-filtered
    )

    # Extract and save training samples
    training_samples = consensus_module.get_high_quality_samples(labeled_trajectories)
    logger.info(f"Extracted {len(training_samples)} training samples")

    if args.output_training_samples:
        output_samples_path = args.output_training_samples
    else:
        output_samples_path = args.output.parent / f"{args.output.stem}_training_samples.jsonl"

    logger.info(f"Saving training samples to {output_samples_path}")
    save_training_samples(
        labeled_trajectories,
        output_samples_path,
        format=config.output.format,
    )

    # Save statistics
    stats_output = args.output.parent / f"{args.output.stem}_stats.json"
    logger.info(f"Saving statistics to {stats_output}")
    with open(stats_output, 'w') as f:
        json.dump({
            "total_trajectories": len(trajectories),
            "filtered_trajectories": sum(lt.is_filtered for lt in labeled_trajectories),
            "kept_trajectories": sum(not lt.is_filtered for lt in labeled_trajectories),
            "total_steps": stats.total_steps,
            "positive_consensus": stats.positive_consensus,
            "negative_consensus": stats.negative_consensus,
            "disagreements": stats.disagreements,
            "agreement_rate": stats.agreement_rate,
            "label_distribution": stats.label_distribution,
            "training_samples": len(training_samples),
        }, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"✓ Processed {len(trajectories)} trajectories")
    logger.info(f"✓ Kept {sum(not lt.is_filtered for lt in labeled_trajectories)} high-quality trajectories")
    logger.info(f"✓ Generated {len(training_samples)} training samples")
    logger.info(f"✓ Agreement rate: {stats.agreement_rate:.1%}")


if __name__ == "__main__":
    main()
