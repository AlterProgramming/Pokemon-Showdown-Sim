#!/usr/bin/env python3
"""
Analyze alignment between decoded past actions and policy decisions.

Metrics:
  - Decoder-policy agreement rate (top-1, top-2)
  - Action type distribution (moves vs switches)
  - Decoder entropy (latent space coverage)
  - Temporal coherence of decoded sequences
"""

import json
import argparse
from collections import Counter
from typing import Dict, List, Tuple
import numpy as np


def load_jsonl(path: str) -> List[dict]:
    """Load JSONL log file."""
    records = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def compute_agreement_rate(records: List[dict]) -> Dict[str, float]:
    """
    Compute how often decoder-predicted actions match the policy's selected action.
    """
    top1_agree = 0
    top2_agree = 0
    total = 0

    for record in records:
        decoded = record.get("decoded_actions_this_turn", [])
        selected = record.get("selected_action_token", "")

        if not decoded or not selected:
            continue

        total += 1

        # Top-1 agreement
        if decoded[0] == selected:
            top1_agree += 1

        # Top-2 agreement
        if len(decoded) >= 2 and selected in decoded[:2]:
            top2_agree += 1

    return {
        "total_turns": total,
        "top1_agreement": top1_agree / total if total > 0 else 0.0,
        "top2_agreement": top2_agree / total if total > 0 else 0.0,
    }


def compute_action_type_distribution(records: List[dict]) -> Dict[str, float]:
    """
    Break down decoded actions by type (move vs switch).
    """
    moves = 0
    switches = 0
    total = 0

    for record in records:
        decoded = record.get("decoded_actions_this_turn", [])
        for action in decoded:
            if action.startswith("move"):
                moves += 1
            elif action.startswith("switch"):
                switches += 1
            total += 1

    return {
        "move_fraction": moves / total if total > 0 else 0.0,
        "switch_fraction": switches / total if total > 0 else 0.0,
        "unknown_fraction": (total - moves - switches) / total if total > 0 else 0.0,
    }


def compute_decoder_entropy(records: List[dict]) -> Dict[str, float]:
    """
    Compute entropy of decoded action logits (latent space diversity).

    High entropy = diverse predictions = healthy latent space.
    Low entropy = collapsed predictions = poor representation.
    """
    all_entropies = []

    for record in records:
        for analysis in record.get("analyses", []):
            logits = analysis.get("decoded_action_logits", [])
            if not logits:
                continue

            # Convert logits to probabilities
            logits = np.array(logits, dtype=np.float32)
            probs = np.exp(logits - np.max(logits))  # Numerical stability
            probs /= probs.sum()

            # Shannon entropy
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            all_entropies.append(entropy)

    if not all_entropies:
        return {"mean_entropy": 0.0, "std_entropy": 0.0, "min_entropy": 0.0, "max_entropy": 0.0}

    return {
        "mean_entropy": float(np.mean(all_entropies)),
        "std_entropy": float(np.std(all_entropies)),
        "min_entropy": float(np.min(all_entropies)),
        "max_entropy": float(np.max(all_entropies)),
    }


def compute_temporal_coherence(records: List[dict]) -> Dict[str, float]:
    """
    Measure whether decoded action sequences form coherent patterns.

    E.g., setup moves (screens, hazards) followed by offense, then switch.
    """
    # Heuristic: count how often "setup" moves (screens, hazards) appear
    # before "offensive" moves in the same decoded sequence

    setup_move_keywords = ["screen", "hazard", "stealth", "reflect", "light screen", "toxic spikes"]
    offensive_keywords = ["boost", "dragon", "earthquake", "bullet punch"]

    coherent_sequences = 0
    total_sequences = 0

    for record in records:
        decoded = record.get("decoded_actions_this_turn", [])
        if len(decoded) < 2:
            continue

        # Check if any setup move appears before an offensive move
        for i, action1 in enumerate(decoded):
            for j in range(i + 1, len(decoded)):
                action2 = decoded[j]
                if any(kw in action1.lower() for kw in setup_move_keywords):
                    if any(kw in action2.lower() for kw in offensive_keywords):
                        coherent_sequences += 1
                        break

        total_sequences += 1

    return {
        "coherent_sequences": coherent_sequences,
        "total_sequences": total_sequences,
        "coherence_rate": coherent_sequences / total_sequences if total_sequences > 0 else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze decoded action predictions and policy alignment"
    )
    parser.add_argument("log", help="Path to JSONL log file")
    parser.add_argument("--output-csv", help="Output CSV file for per-turn analysis")
    args = parser.parse_args()

    print(f"Loading {args.log}...")
    records = load_jsonl(args.log)
    print(f"Loaded {len(records)} records\n")

    # Compute metrics
    agreement = compute_agreement_rate(records)
    action_types = compute_action_type_distribution(records)
    entropy = compute_decoder_entropy(records)
    coherence = compute_temporal_coherence(records)

    # Print summary
    print("=" * 60)
    print("DECODER-POLICY ALIGNMENT")
    print("=" * 60)
    print(f"Top-1 agreement rate: {agreement['top1_agreement']:.1%}")
    print(f"Top-2 agreement rate: {agreement['top2_agreement']:.1%}")
    print(f"Total turns analyzed: {agreement['total_turns']}\n")

    print("=" * 60)
    print("ACTION TYPE DISTRIBUTION (decoded)")
    print("=" * 60)
    print(f"Move actions: {action_types['move_fraction']:.1%}")
    print(f"Switch actions: {action_types['switch_fraction']:.1%}")
    print(f"Unknown/other: {action_types['unknown_fraction']:.1%}\n")

    print("=" * 60)
    print("LATENT SPACE HEALTH (decoder entropy)")
    print("=" * 60)
    print(f"Mean entropy: {entropy['mean_entropy']:.2f} bits")
    print(f"Std entropy: {entropy['std_entropy']:.2f} bits")
    print(f"Range: [{entropy['min_entropy']:.2f}, {entropy['max_entropy']:.2f}] bits")
    print("(Higher = more diverse; ~2.0-2.5 is healthy for ~10 actions)\n")

    print("=" * 60)
    print("TEMPORAL COHERENCE (action sequences)")
    print("=" * 60)
    print(f"Coherent sequences: {coherence['coherent_sequences']} / {coherence['total_sequences']}")
    print(f"Coherence rate: {coherence['coherence_rate']:.1%}\n")

    if args.output_csv:
        print(f"CSV output not yet implemented (path: {args.output_csv})")


if __name__ == "__main__":
    main()
