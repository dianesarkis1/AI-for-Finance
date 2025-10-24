"""
Run batch evaluation on a sample from train.jsonl.

This script loads a specific sample from the training data and evaluates it
using the batch API for faster processing.

Usage:
    python example_usage.py --index 0                    # Evaluate first sample
    python example_usage.py --index 12                   # Evaluate sample at index 12
    python example_usage.py --resume batch_abc123        # Resume interrupted batch
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evals.batch_evals import evaluate_memo_batch, resume_batch_evaluation


def load_sample_from_train(train_file: str, index: int):
    """
    Load a specific sample from train.jsonl by index.

    Args:
        train_file: Path to train.jsonl
        index: Zero-based index of sample to load

    Returns:
        Tuple of (memo, source_document)
    """
    with open(train_file, "r") as f:
        for i, line in enumerate(f):
            if i == index:
                sample = json.loads(line)
                messages = sample.get("messages", [])

                system_msg = next((m for m in messages if m["role"] == "system"), None)
                assistant_msg = next((m for m in messages if m["role"] == "assistant"), None)

                if system_msg and assistant_msg:
                    source_document = system_msg["content"]
                    memo = assistant_msg["content"]
                    return memo, source_document
                else:
                    raise ValueError(f"Sample at index {index} missing system or assistant message")

    raise ValueError(f"Index {index} not found in train.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Run batch evaluation on train.jsonl samples")
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of sample to evaluate from train.jsonl (default: 0)"
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="../../data/train.jsonl",
        help="Path to train.jsonl file (default: ../../data/train.jsonl)"
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Seconds between status checks (default: 60)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume monitoring a batch job by batch_id (e.g., batch_abc123)"
    )

    args = parser.parse_args()

    # Handle resume case
    if args.resume:
        print("=" * 70)
        print(f"RESUMING BATCH JOB: {args.resume}")
        print("=" * 70)

        try:
            results = resume_batch_evaluation(
                batch_id=args.resume,
                poll_interval=args.poll_interval
            )
            print(f"\n✅ Received {len(results)} results from batch\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            sys.exit(1)

        return

    # Normal evaluation case
    print("=" * 70)
    print("BATCH EVALUATION ON TRAIN.JSONL")
    print("=" * 70)
    print(f"Train file: {args.train_file}")
    print(f"Sample index: {args.index}")
    print(f"Poll interval: {args.poll_interval}s")
    print("=" * 70)

    # Load sample
    try:
        memo, source_document = load_sample_from_train(args.train_file, args.index)
        print(f"\n✅ Loaded sample {args.index}")
        print(f"   Source document: {len(source_document)} chars")
        print(f"   Memo: {len(memo)} chars\n")
    except FileNotFoundError:
        print(f"\n❌ Error: Training file not found: {args.train_file}")
        print(f"   Make sure to run from the correct directory or specify --train-file\n")
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ Error: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error loading sample: {e}\n")
        sys.exit(1)

    # Evaluate using batch API
    try:
        score = evaluate_memo_batch(
            memo=memo,
            source_document=source_document,
            model="gpt-5",
            poll_interval=args.poll_interval
        )

        print(f"\n{'='*70}")
        print(f"✅ FINAL SCORE: {score:.2f}/100")
        print(f"{'='*70}\n")

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
