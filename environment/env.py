"""
Graph-RCA Pipeline Diagnoser — OpenEnv Environment

An RL environment where an AI agent diagnoses root causes in failing data
pipelines by traversing a DAG of pipeline nodes and inspecting logs.

Built on 200 real annotated production incidents from companies including
Cloudflare, AWS, GitHub, Google, and more.

OpenEnv API:
  reset(task_id) -> StepResult
  step(action)   -> StepResult
  state()        -> dict
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from environment.grader import grade_diagnosis, grade_step_reward
from environment.models import (
    ActionType,
    LogEntry,
    NodeInfo,
    PipelineAction,
    PipelineObservation,
    PipelineReward,
)


# ---------------------------------------------------------------------------
# StepResult — returned by reset() and step()
# ---------------------------------------------------------------------------

class StepResult:
    """Container for the result of a reset() or step() call."""

    def __init__(
        self,
        observation: PipelineObservation,
        reward: float,
        done: bool,
        info: Dict[str, Any],
    ):
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation": self.observation.model_dump(),
            "reward": self.reward,
            "done": self.done,
            "info": self.info,
        }


# ---------------------------------------------------------------------------
# Task loader
# ---------------------------------------------------------------------------

TASKS_DIR = Path(__file__).parent / "tasks"

TASK_MAP = {
    "single_point_failure": "task_easy.json",
    "cascading_failure": "task_medium.json",
    "simultaneous_failures": "task_hard.json",
}


def _load_task(task_id: str) -> Dict[str, Any]:
    """Load a task definition from JSON."""
    filename = TASK_MAP.get(task_id)
    if filename is None:
        raise ValueError(
            f"Unknown task_id '{task_id}'. "
            f"Available tasks: {list(TASK_MAP.keys())}"
        )
    task_path = TASKS_DIR / filename
    with open(task_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main environment class
# ---------------------------------------------------------------------------

class GraphRCAEnv:
    """
    Graph-RCA Pipeline Diagnoser OpenEnv Environment.

    The agent must:
    1. Explore the pipeline DAG by inspecting node logs
    2. Identify which node(s) are the root cause of failures
    3. Submit a diagnosis with the root cause node IDs

    Reward signal:
    - Dense exploration rewards for inspecting relevant nodes
    - F1-based final reward for correct root cause identification
    - Efficiency bonus for solving quickly
    - Small step penalty to discourage random wandering
    """

    AVAILABLE_TASKS = list(TASK_MAP.keys())

    def __init__(self) -> None:
        self._task_data: Optional[Dict[str, Any]] = None
        self._task_id: Optional[str] = None
        self._current_step: int = 0
        self._done: bool = False
        self._nodes_inspected: List[str] = []
        self._diagnosis_submitted: bool = False
        self._last_reward: float = 0.0

        # Indexed for O(1) lookup
        self._node_lookup: Dict[str, Dict[str, Any]] = {}

    # -----------------------------------------------------------------------
    # OpenEnv API
    # -----------------------------------------------------------------------

    def reset(self, task_id: Optional[str] = None) -> StepResult:
        """
        Reset the environment with a new task.

        Args:
            task_id: One of 'single_point_failure', 'cascading_failure',
                     'simultaneous_failures'. Defaults to 'single_point_failure'.

        Returns:
            StepResult with initial observation, reward=0.0, done=False.
        """
        if task_id is None:
            task_id = "single_point_failure"

        self._task_data = _load_task(task_id)
        self._task_id = task_id
        self._current_step = 0
        self._done = False
        self._nodes_inspected = []
        self._diagnosis_submitted = False
        self._last_reward = 0.0

        # Build node lookup index
        self._node_lookup = {}
        for node in self._task_data["dag_structure"]["nodes"]:
            self._node_lookup[node["id"]] = node

        obs = self._build_observation(
            last_action=None,
            last_action_result=f"Environment reset. Task: {task_id}. "
                               f"Difficulty: {self._task_data.get('difficulty', 'unknown')}. "
                               f"Inspect pipeline nodes to identify root cause(s).",
        )

        return StepResult(observation=obs, reward=0.0, done=False, info={"task_id": task_id})

    def step(self, action: PipelineAction) -> StepResult:
        """
        Execute one action in the environment.

        Args:
            action: PipelineAction specifying what to do.

        Returns:
            StepResult with updated observation, reward, done flag, and info.
        """
        if self._task_data is None:
            raise RuntimeError("Call reset() before step()")

        if self._done:
            obs = self._build_observation(
                last_action=action.action_type.value,
                last_action_result="Episode already finished. Call reset() to start a new episode.",
            )
            return StepResult(observation=obs, reward=0.0, done=True, info={"episode_done": True})

        self._current_step += 1
        max_steps = self._task_data.get("max_steps", 15)

        reward = 0.0
        done = False
        info: Dict[str, Any] = {}

        # Execute action
        if action.action_type == ActionType.LIST_NODES:
            result_msg, reward = self._action_list_nodes()

        elif action.action_type == ActionType.INSPECT_NODE:
            result_msg, reward = self._action_inspect_node(action.target_node)

        elif action.action_type == ActionType.TRAVERSE_EDGE:
            result_msg, reward = self._action_traverse_edge(action.target_node)

        elif action.action_type == ActionType.GET_METRICS:
            result_msg, reward = self._action_get_metrics()

        elif action.action_type in (ActionType.DIAGNOSE, ActionType.SUBMIT_DIAGNOSIS):
            result_msg, reward, done, info = self._action_diagnose(
                action.diagnosis, max_steps
            )

        else:
            result_msg = f"Unknown action type: {action.action_type}"
            reward = -0.02

        # Check if we've run out of steps (only if not already done from diagnosis)
        if not done and self._current_step >= max_steps:
            done = True
            self._done = True
            result_msg += f" [Episode ended: {max_steps} steps exhausted without diagnosis]"
            info["timeout"] = True

        self._last_reward = reward

        obs = self._build_observation(
            last_action=action.action_type.value,
            last_action_result=result_msg,
        )
        obs.diagnosis_submitted = self._diagnosis_submitted

        return StepResult(observation=obs, reward=reward, done=done, info=info)

    def state(self) -> Dict[str, Any]:
        """
        Return the current environment state as a dict.

        Useful for debugging and logging.
        """
        if self._task_data is None:
            return {"status": "not_initialized"}

        return {
            "task_id": self._task_id,
            "difficulty": self._task_data.get("difficulty"),
            "current_step": self._current_step,
            "max_steps": self._task_data.get("max_steps"),
            "done": self._done,
            "nodes_inspected": self._nodes_inspected,
            "diagnosis_submitted": self._diagnosis_submitted,
            "last_reward": self._last_reward,
            "node_count": len(self._task_data["dag_structure"]["nodes"]),
            "edge_count": len(self._task_data["dag_structure"]["edges"]),
        }

    # -----------------------------------------------------------------------
    # Action implementations
    # -----------------------------------------------------------------------

    def _action_list_nodes(self) -> Tuple[str, float]:
        """List all nodes with their current status."""
        nodes = self._task_data["dag_structure"]["nodes"]
        node_status = self._task_data["node_status"]

        lines = ["Pipeline DAG nodes:"]
        for node in nodes:
            nid = node["id"]
            status = node_status.get(nid, "UNKNOWN")
            logs = self._task_data["node_logs"].get(nid, [])
            error_levels = {"ERROR", "CRITICAL", "FATAL"}
            has_errors = any(e["level"] in error_levels for e in logs)
            flag = " [!]" if has_errors else ""
            lines.append(f"  {nid} ({node['label']}) [{status}]{flag} — {len(logs)} logs")

        # Tiny positive reward for taking an exploratory action
        reward = grade_step_reward(
            self._nodes_inspected,
            self._task_data["ground_truth"]["root_cause_nodes"],
            len(nodes),
            self._current_step,
            self._task_data.get("max_steps", 15),
        )
        return "\n".join(lines), reward

    def _action_inspect_node(self, node_id: Optional[str]) -> Tuple[str, float]:
        """Inspect logs of a specific node."""
        if not node_id:
            return "inspect_node requires target_node to be set.", -0.02

        if node_id not in self._node_lookup:
            valid = list(self._node_lookup.keys())
            return f"Node '{node_id}' not found. Valid nodes: {valid}", -0.02

        if node_id not in self._nodes_inspected:
            self._nodes_inspected.append(node_id)

        logs = self._task_data["node_logs"].get(node_id, [])
        node_info = self._node_lookup[node_id]
        status = self._task_data["node_status"].get(node_id, "UNKNOWN")

        lines = [f"Node: {node_id} ({node_info['label']}) | Status: {status} | Type: {node_info['type']}"]
        if logs:
            lines.append("Logs:")
            for entry in logs:
                lines.append(f"  [{entry['level']}] {entry['timestamp']} — {entry['message']}")
        else:
            lines.append("No logs available for this node.")

        reward = grade_step_reward(
            self._nodes_inspected,
            self._task_data["ground_truth"]["root_cause_nodes"],
            len(self._task_data["dag_structure"]["nodes"]),
            self._current_step,
            self._task_data.get("max_steps", 15),
        )
        return "\n".join(lines), reward

    def _action_traverse_edge(self, target_node: Optional[str]) -> Tuple[str, float]:
        """Show edges connected to a node (neighbors)."""
        if not target_node:
            return "traverse_edge requires target_node to be set.", -0.02

        if target_node not in self._node_lookup:
            return f"Node '{target_node}' not found.", -0.02

        edges = self._task_data["dag_structure"]["edges"]
        incoming = [e["from"] for e in edges if e["to"] == target_node]
        outgoing = [e["to"] for e in edges if e["from"] == target_node]

        lines = [f"Edges for node: {target_node}"]
        if incoming:
            lines.append(f"  Upstream (parents): {incoming}")
        else:
            lines.append("  Upstream (parents): none — this is a root node")
        if outgoing:
            lines.append(f"  Downstream (children): {outgoing}")
        else:
            lines.append("  Downstream (children): none — this is a leaf node")

        reward = grade_step_reward(
            self._nodes_inspected,
            self._task_data["ground_truth"]["root_cause_nodes"],
            len(self._task_data["dag_structure"]["nodes"]),
            self._current_step,
            self._task_data.get("max_steps", 15),
        )
        return "\n".join(lines), reward

    def _action_get_metrics(self) -> Tuple[str, float]:
        """Return pipeline metrics and data samples."""
        metrics = self._task_data.get("data_samples", {})
        if metrics:
            lines = ["Pipeline metrics:"]
            for category, values in metrics.items():
                lines.append(f"  {category}:")
                if isinstance(values, dict):
                    for k, v in values.items():
                        lines.append(f"    {k}: {v}")
                else:
                    lines.append(f"    {values}")
        else:
            lines = ["No metrics data available for this task."]

        reward = grade_step_reward(
            self._nodes_inspected,
            self._task_data["ground_truth"]["root_cause_nodes"],
            len(self._task_data["dag_structure"]["nodes"]),
            self._current_step,
            self._task_data.get("max_steps", 15),
        )
        return "\n".join(lines), reward

    def _action_diagnose(
        self,
        diagnosis,
        max_steps: int,
    ) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Grade a diagnosis and end the episode."""
        if not diagnosis:
            return (
                "diagnose requires a diagnosis object with 'root_cause_nodes' and 'failure_type'.",
                -0.02, False, {},
            )

        self._diagnosis_submitted = True
        self._done = True

        ground_truth = self._task_data["ground_truth"]
        reward_obj: PipelineReward = grade_diagnosis(
            predicted_nodes=diagnosis.root_cause_nodes,
            ground_truth_nodes=ground_truth["root_cause_nodes"],
            ground_truth_keywords=ground_truth.get("keywords", []),
            steps_used=self._current_step,
            max_steps=max_steps,
            nodes_inspected=self._nodes_inspected,
            total_nodes=len(self._task_data["dag_structure"]["nodes"]),
            explanation=diagnosis.failure_type,
        )

        info = {
            "score": reward_obj.score,
            "precision": reward_obj.precision,
            "recall": reward_obj.recall,
            "f1": reward_obj.f1,
            "efficiency_bonus": reward_obj.efficiency_bonus,
            "exploration_reward": reward_obj.exploration_reward,
            "is_correct": reward_obj.is_correct,
            "feedback": reward_obj.feedback,
            "ground_truth": ground_truth["root_cause_nodes"],
        }

        return reward_obj.feedback, reward_obj.score, True, info

    # -----------------------------------------------------------------------
    # Observation builder
    # -----------------------------------------------------------------------

    def _build_observation(
        self,
        last_action: Optional[str],
        last_action_result: Optional[str],
        inspected_node: Optional[str] = None,
    ) -> PipelineObservation:
        """Build a PipelineObservation from the current environment state."""
        if self._task_data is None:
            raise RuntimeError("No task loaded")

        node_status = self._task_data["node_status"]
        all_nodes = self._task_data["dag_structure"]["nodes"]
        error_levels = {"ERROR", "CRITICAL", "FATAL"}

        nodes_info: List[NodeInfo] = []
        for n in all_nodes:
            nid = n["id"]
            logs = self._task_data["node_logs"].get(nid, [])
            has_errors = any(e["level"] in error_levels for e in logs)
            nodes_info.append(NodeInfo(
                node_id=nid,
                label=n["label"],
                node_type=n["type"],
                status=node_status.get(nid, "UNKNOWN"),
                log_count=len(logs),
                has_errors=has_errors,
            ))

        max_steps = self._task_data.get("max_steps", 15)

        # Populate inspected logs if the last action was an inspect
        inspected_logs: List[LogEntry] = []
        current_inspected: Optional[str] = None
        if last_action == ActionType.INSPECT_NODE.value and self._nodes_inspected:
            current_inspected = self._nodes_inspected[-1]
            raw_logs = self._task_data["node_logs"].get(current_inspected, [])
            inspected_logs = [
                LogEntry(
                    timestamp=e["timestamp"],
                    level=e["level"],
                    message=e["message"],
                )
                for e in raw_logs
            ]

        # Include metrics if last action was get_metrics
        metrics = None
        if last_action == ActionType.GET_METRICS.value:
            metrics = self._task_data.get("data_samples")

        return PipelineObservation(
            task_id=self._task_id or "",
            difficulty=self._task_data.get("difficulty", "unknown"),
            nodes=nodes_info,
            edges=self._task_data["dag_structure"]["edges"],
            inspected_node=current_inspected,
            inspected_logs=inspected_logs,
            metrics=metrics,
            current_step=self._current_step,
            max_steps=max_steps,
            steps_remaining=max(0, max_steps - self._current_step),
            last_action=last_action,
            last_action_result=last_action_result,
            nodes_inspected=list(self._nodes_inspected),
            diagnosis_submitted=self._diagnosis_submitted,
        )
