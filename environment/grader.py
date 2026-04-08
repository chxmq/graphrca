"""
Deterministic grader for the Graph-RCA Pipeline Diagnoser environment.

Scores agent diagnoses against ground truth root cause nodes.
All scoring is purely deterministic — no randomness, no LLM calls.

Score formula:
  final_score = (f1_score * 0.80) + (efficiency_bonus * 0.20)

where:
  f1_score  = harmonic mean of precision and recall over predicted vs true root cause nodes
  efficiency_bonus = reward for fewer steps used (normalized to max_steps)
"""

from __future__ import annotations

import re
from typing import List, Set, Tuple

from environment.models import PipelineReward


def parse_diagnosis(diagnosis: str) -> Tuple[List[str], str]:
    """
    Parse agent's diagnosis string into node IDs and explanation.

    Accepts formats:
      "node_id1"
      "node_id1, node_id2"
      "node_id1 | explanation text"
      "node_id1, node_id2 | explanation"

    Returns:
        (list_of_node_ids, explanation_text)
    """
    if "|" in diagnosis:
        nodes_part, explanation = diagnosis.split("|", 1)
    else:
        nodes_part = diagnosis
        explanation = ""

    # Extract node IDs: split by comma or space, filter to valid node_* patterns
    raw = [t.strip() for t in re.split(r"[,\s]+", nodes_part.strip()) if t.strip()]

    # Accept any token that starts with "node_" or looks like a valid identifier
    node_ids = [t for t in raw if t]

    return node_ids, explanation.strip()


def compute_f1(predicted: Set[str], ground_truth: Set[str]) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 for set-based classification.

    Returns:
        (precision, recall, f1)
    """
    if not predicted and not ground_truth:
        return 1.0, 1.0, 1.0

    if not predicted or not ground_truth:
        return 0.0, 0.0, 0.0

    true_positives = len(predicted & ground_truth)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(ground_truth) if ground_truth else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1


def compute_keyword_bonus(explanation: str, keywords: List[str]) -> float:
    """
    Small bonus (up to 0.05) for mentioning relevant keywords in explanation.
    Encourages agents to provide meaningful diagnoses, not just node IDs.
    """
    if not explanation or not keywords:
        return 0.0

    explanation_lower = explanation.lower()
    matched = sum(1 for kw in keywords if kw.lower() in explanation_lower)
    return min(0.05, 0.01 * matched)


def compute_efficiency_bonus(steps_used: int, max_steps: int) -> float:
    """
    Efficiency bonus for solving the task quickly.

    Bonus ranges from 0.0 (all steps used) to 0.2 (minimal steps used).
    Formula: 0.2 * (1 - steps_used / max_steps) but only if diagnosis is correct.
    """
    if max_steps <= 0:
        return 0.0
    ratio = steps_used / max_steps
    return round(max(0.0, 0.2 * (1.0 - ratio)), 4)


def compute_exploration_reward(
    nodes_inspected: List[str],
    root_cause_nodes: List[str],
    total_nodes: int,
) -> float:
    """
    Small reward for inspecting root cause nodes during exploration.
    Provides dense signal even before diagnosis is submitted.

    Max exploration reward: 0.1
    """
    if not root_cause_nodes or total_nodes == 0:
        return 0.0

    inspected_set = set(nodes_inspected)
    rc_set = set(root_cause_nodes)

    # Fraction of root cause nodes that were inspected
    if not rc_set:
        return 0.0

    fraction_explored = len(inspected_set & rc_set) / len(rc_set)
    return round(fraction_explored * 0.1, 4)


def grade_diagnosis(
    predicted_nodes: List[str],
    ground_truth_nodes: List[str],
    ground_truth_keywords: List[str],
    steps_used: int,
    max_steps: int,
    nodes_inspected: List[str],
    total_nodes: int,
    explanation: str = "",
) -> PipelineReward:
    """
    Grade a submitted diagnosis against ground truth.

    This is the primary grading function — fully deterministic.

    Args:
        predicted_nodes: Node IDs identified by the agent as root causes
        ground_truth_nodes: True root cause node IDs
        ground_truth_keywords: Keywords that should appear in a good explanation
        steps_used: Number of steps taken so far
        max_steps: Maximum steps allowed
        nodes_inspected: All nodes the agent inspected
        total_nodes: Total number of nodes in the DAG
        explanation: Optional free-text explanation (used for keyword bonus)

    Returns:
        PipelineReward with all scoring components filled in
    """
    predicted_set = set(predicted_nodes)
    truth_set = set(ground_truth_nodes)

    precision, recall, f1 = compute_f1(predicted_set, truth_set)
    is_correct = predicted_set == truth_set

    # Efficiency only counts when the diagnosis is at least partially correct
    if f1 > 0:
        efficiency = compute_efficiency_bonus(steps_used, max_steps)
    else:
        efficiency = 0.0

    # Keyword bonus (capped at 0.05, only if f1 > 0)
    keyword_bonus = compute_keyword_bonus(explanation, ground_truth_keywords) if f1 > 0 else 0.0

    # Exploration reward (always given — dense signal during episode)
    exploration = compute_exploration_reward(nodes_inspected, ground_truth_nodes, total_nodes)

    # Final score formula
    base_score = (f1 * 0.80) + (efficiency * 0.20)
    total_score = min(1.0, base_score + keyword_bonus + exploration)

    # Build feedback message
    if is_correct:
        feedback = (
            f"Correct! Identified all root cause nodes: {sorted(truth_set)}. "
            f"F1={f1:.2f}, efficiency_bonus={efficiency:.2f}"
        )
    elif f1 > 0:
        missed = truth_set - predicted_set
        extra = predicted_set - truth_set
        parts = []
        if missed:
            parts.append(f"missed nodes: {sorted(missed)}")
        if extra:
            parts.append(f"extra nodes: {sorted(extra)}")
        feedback = f"Partial credit. {'; '.join(parts)}. F1={f1:.2f}"
    else:
        feedback = (
            f"Incorrect. Predicted: {sorted(predicted_set)}, "
            f"Expected: {sorted(truth_set)}. Score=0."
        )

    return PipelineReward(
        score=round(total_score, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        efficiency_bonus=round(efficiency, 4),
        exploration_reward=round(exploration, 4),
        is_correct=is_correct,
        is_terminal=True,
        feedback=feedback,
    )


def grade_step_reward(
    nodes_inspected: List[str],
    root_cause_nodes: List[str],
    total_nodes: int,
    steps_used: int,
    max_steps: int,
) -> float:
    """
    Per-step reward for non-diagnosis actions (exploration signal).

    Provides a small positive reward when the agent inspects root cause nodes,
    and a tiny negative penalty for wasting steps.

    Returns a float reward in range [0.0, 0.1].
    """
    exploration = compute_exploration_reward(nodes_inspected, root_cause_nodes, total_nodes)

    # Small step penalty to discourage random wandering (clamped to keep reward >= 0)
    step_penalty = -0.01 * (steps_used / max_steps)

    return round(max(0.0, exploration + step_penalty), 4)
