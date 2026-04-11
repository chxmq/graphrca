"""OpenEnv typed models for the Graph-RCA Pipeline Diagnoser environment."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action types the agent can perform
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    INSPECT_NODE = "inspect_node"
    TRAVERSE_EDGE = "traverse_edge"
    SUBMIT_DIAGNOSIS = "submit_diagnosis"
    LIST_NODES = "list_nodes"
    GET_METRICS = "get_metrics"


# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------

class NodeInfo(BaseModel):
    """Information about a single DAG node visible to the agent."""
    node_id: str = Field(description="Unique identifier of the node")
    label: str = Field(description="Human-readable label for the node")
    node_type: str = Field(description="Type of node (service, network, monitoring, etc.)")
    status: str = Field(description="Current status: HEALTHY, DEGRADED, or FAILED")
    log_count: int = Field(description="Number of log entries in this node")
    has_errors: bool = Field(description="True if node has ERROR or CRITICAL logs")


class LogEntry(BaseModel):
    """A single log entry from a pipeline node."""
    timestamp: str = Field(description="ISO 8601 timestamp")
    level: str = Field(description="Log level: INFO, WARNING, ERROR, CRITICAL")
    message: str = Field(description="Log message content")


class PipelineObservation(BaseModel):
    """
    What the agent sees at each step.

    Contains the DAG topology, current node status, and inspected log content.
    """
    task_id: str = Field(description="Current task identifier")
    difficulty: str = Field(description="Task difficulty: easy, medium, or hard")

    # DAG topology
    nodes: List[NodeInfo] = Field(description="All nodes in the pipeline DAG")
    edges: List[Dict[str, str]] = Field(description="Edges as list of {from, to} dicts")

    # Inspection results (populated after inspect_node action)
    inspected_node: Optional[str] = Field(None, description="Node ID currently being inspected")
    inspected_logs: List[LogEntry] = Field(default_factory=list, description="Logs from inspected node")

    # Metrics (populated after get_metrics action)
    metrics: Optional[Dict[str, Any]] = Field(None, description="Pipeline metrics and data samples")

    # Episode progress
    current_step: int = Field(description="Current step number (1-indexed)")
    max_steps: int = Field(description="Maximum steps allowed for this task")
    steps_remaining: int = Field(description="Steps left before episode ends")

    # Action feedback
    last_action: Optional[str] = Field(None, description="Last action taken by agent")
    last_action_result: Optional[str] = Field(None, description="Result/feedback from last action")

    # Diagnosis tracking
    nodes_inspected: List[str] = Field(default_factory=list, description="All nodes the agent has inspected")
    diagnosis_submitted: bool = Field(False, description="Whether agent has submitted a diagnosis")


class DiagnosisInput(BaseModel):
    """Structured diagnosis submitted by the agent."""
    root_cause_nodes: List[str] = Field(
        description="List of node IDs identified as root causes"
    )
    failure_type: str = Field(
        description="Short description of the failure type (e.g. 'service_crash', 'misconfiguration')"
    )


class PipelineAction(BaseModel):
    """
    An action the agent can take in the environment.

    action_type controls which fields are used:
    - inspect_node: target_node required
    - traverse_edge: target_node required
    - diagnose / submit_diagnosis: diagnosis required (structured object)
    - list_nodes: no additional fields needed
    - get_metrics: no additional fields needed
    """
    action_type: ActionType = Field(description="Type of action to perform")
    target_node: Optional[str] = Field(None, description="Node ID to inspect or traverse to")
    diagnosis: Optional[DiagnosisInput] = Field(
        None,
        description=(
            "Structured root cause diagnosis with 'root_cause_nodes' (list of node IDs) "
            "and 'failure_type' (string). "
            "Example: {\"root_cause_nodes\": [\"node_cluster_manager\"], \"failure_type\": \"service_crash\"}"
        )
    )


class PipelineReward(BaseModel):
    """
    Reward signal for the current step.

    Provides dense reward signal throughout the episode, not just at the end.
    """
    score: float = Field(description="Total reward for this step [0.0-1.0]")

    # Partial progress components
    precision: float = Field(0.0, description="Fraction of predicted root causes that are correct")
    recall: float = Field(0.0, description="Fraction of true root causes that were predicted")
    f1: float = Field(0.0, description="F1 score of root cause identification")

    efficiency_bonus: float = Field(0.0, description="Bonus for diagnosing in fewer steps [0.0-0.2]")
    exploration_reward: float = Field(0.0, description="Small reward for inspecting relevant nodes")

    is_correct: bool = Field(False, description="True if diagnosis perfectly matches ground truth")
    is_terminal: bool = Field(False, description="True if episode ended this step")

    feedback: str = Field("", description="Human-readable feedback on the diagnosis")
