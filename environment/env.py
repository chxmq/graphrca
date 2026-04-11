"""
Graph-RCA Pipeline Diagnoser — OpenEnv Environment

An RL environment where an AI agent diagnoses root causes in failing data
pipelines by traversing a DAG of pipeline nodes and inspecting logs.
"""

import json
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

_ERROR_LEVELS = frozenset({"ERROR", "CRITICAL", "FATAL"})

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

TASKS_DIR = Path(__file__).parent / "tasks"

TASK_MAP = {
    "single_point_failure": "task_easy.json",
    "cascading_failure": "task_medium.json",
    "simultaneous_failures": "task_hard.json",
}


def _load_task(task_id: str) -> Dict[str, Any]:
    filename = TASK_MAP.get(task_id)
    if filename is None:
        raise ValueError(
            f"Unknown task_id '{task_id}'. "
            f"Available tasks: {list(TASK_MAP.keys())}"
        )
    with open(TASKS_DIR / filename, "r", encoding="utf-8") as f:
        return json.load(f)


class GraphRCAEnv:

    AVAILABLE_TASKS = list(TASK_MAP.keys())

    def __init__(self) -> None:
        self._task_data: Optional[Dict[str, Any]] = None
        self._task_id: Optional[str] = None
        self._current_step: int = 0
        self._done: bool = False
        self._nodes_inspected: List[str] = []
        self._diagnosis_submitted: bool = False
        self._last_reward: float = 0.0
        self._node_lookup: Dict[str, Dict[str, Any]] = {}

    def reset(self, task_id: Optional[str] = None) -> StepResult:
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

        elif action.action_type == ActionType.SUBMIT_DIAGNOSIS:
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

    def _action_list_nodes(self) -> Tuple[str, float]:
        nodes = self._task_data["dag_structure"]["nodes"]
        node_status = self._task_data["node_status"]

        lines = ["Pipeline DAG nodes:"]
        for node in nodes:
            nid = node["id"]
            status = node_status.get(nid, "UNKNOWN")
            logs = self._task_data["node_logs"].get(nid, [])
            has_errors = any(e["level"] in _ERROR_LEVELS for e in logs)
            flag = " [!]" if has_errors else ""
            lines.append(f"  {nid} ({node['label']}) [{status}]{flag} — {len(logs)} logs")

        reward = grade_step_reward(
            self._nodes_inspected,
            self._task_data["ground_truth"]["root_cause_nodes"],
            len(nodes),
            self._current_step,
            self._task_data.get("max_steps", 15),
        )
        return "\n".join(lines), reward

    def _action_inspect_node(self, node_id: Optional[str]) -> Tuple[str, float]:
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

    def _build_observation(
        self,
        last_action: Optional[str],
        last_action_result: Optional[str],
    ) -> PipelineObservation:
        if self._task_data is None:
            raise RuntimeError("No task loaded")

        node_status = self._task_data["node_status"]
        all_nodes = self._task_data["dag_structure"]["nodes"]

        nodes_info: List[NodeInfo] = []
        for n in all_nodes:
            nid = n["id"]
            logs = self._task_data["node_logs"].get(nid, [])
            has_errors = any(e["level"] in _ERROR_LEVELS for e in logs)
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
