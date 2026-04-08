"""
Graph-RCA Pipeline Diagnoser — Baseline Inference Script

Runs an LLM agent against all 3 tasks in the Graph-RCA OpenEnv environment
using the OpenAI API client.

Environment variables required:
  API_BASE_URL  - The API endpoint for the LLM (e.g. https://api.openai.com/v1)
  MODEL_NAME    - The model identifier (e.g. gpt-4o-mini)
  HF_TOKEN      - Your HuggingFace / API key

Log format (strict — do not deviate):
  [START] task=<task_id> env=graph-rca-pipeline-diagnoser model=<model>
  [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...,rn>

Usage:
  API_BASE_URL=https://api.openai.com/v1 MODEL_NAME=gpt-4o-mini HF_TOKEN=sk-... python inference.py
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN: Optional[str] = os.environ.get("HF_TOKEN")
API_KEY: str = HF_TOKEN or ""  # validated at runtime in main()

ENV_BASE_URL: str = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK: str = "graph-rca-pipeline-diagnoser"

TEMPERATURE: float = 0.2
MAX_TOKENS: int = 512

# Task configs
TASKS = [
    {"task_id": "single_point_failure", "max_steps": 10, "max_total_reward": 1.0},
    {"task_id": "cascading_failure", "max_steps": 15, "max_total_reward": 1.0},
    {"task_id": "simultaneous_failures", "max_steps": 20, "max_total_reward": 1.0},
]

SUCCESS_SCORE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Logging helpers (strict format — do not modify field names)
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    done_str = "true" if done else "false"
    error_str = "null" if error is None else error
    print(
        f"[STEP]  step={step} action={action} reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END]   success={success_str} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompt and prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer (SRE) diagnosing failures in production data pipelines.

You interact with a Graph-RCA environment. Your ONLY goal is to identify the root cause node(s) and submit a diagnosis as fast as possible.

Available actions (respond with EXACTLY one JSON object):
- List all nodes: {"action_type": "list_nodes"}
- Inspect a node's logs: {"action_type": "inspect_node", "target_node": "node_id"}
- Traverse edges: {"action_type": "traverse_edge", "target_node": "node_id"}
- Get metrics: {"action_type": "get_metrics"}
- Submit diagnosis: {"action_type": "submit_diagnosis", "diagnosis": {"root_cause_nodes": ["node_id1"], "failure_type": "brief_failure_type"}}

CRITICAL RULES:
1. You MUST submit a diagnosis using "submit_diagnosis" — this is the ONLY way to score points
2. Inspecting nodes gives tiny rewards but NO final score — you MUST diagnose to win
3. Submit your best diagnosis within 5 steps maximum — do not waste steps re-inspecting
4. If steps_remaining <= 2, you MUST submit a diagnosis immediately — do not inspect
5. The "diagnosis" field MUST be a JSON object with "root_cause_nodes" (array of strings) and "failure_type" (string)

Strategy (fast):
- Step 1: list_nodes → identify FAILED/DEGRADED nodes with [!] error flags
- Step 2-3: inspect the most suspicious FAILED node(s)
- Step 4: submit_diagnosis — pick the FAILED node(s) with CRITICAL/ERROR logs as root cause
- For cascading failures: the root cause is UPSTREAM (the node that caused others to fail)
- For multiple failures: include both node IDs in the root_cause_nodes array

Diagnosis example (single root cause):
{"action_type": "submit_diagnosis", "diagnosis": {"root_cause_nodes": ["node_cluster_manager"], "failure_type": "service_crash"}}

Diagnosis example (multiple root causes):
{"action_type": "submit_diagnosis", "diagnosis": {"root_cause_nodes": ["node_db", "node_cache"], "failure_type": "simultaneous_failures"}}

Respond with ONLY a valid JSON object. No text outside the JSON."""


def build_user_prompt(
    step: int,
    last_observation: Dict[str, Any],
    last_reward: float,
    history: List[str],
) -> str:
    obs = last_observation.get("observation", {})
    task_id = obs.get("task_id", "unknown")
    difficulty = obs.get("difficulty", "unknown")
    steps_remaining = obs.get("steps_remaining", 0)
    last_result = obs.get("last_action_result", "")
    nodes_inspected = obs.get("nodes_inspected", [])

    # Build compact node summary
    nodes = obs.get("nodes", [])
    node_summary = []
    for n in nodes:
        status = n.get("status", "?")
        has_err = "[!]" if n.get("has_errors") else ""
        node_summary.append(f"  {n['node_id']} ({n['label']}) [{status}]{has_err}")

    urgency = ""
    if steps_remaining <= 2:
        urgency = f"\n⚠️ URGENT: Only {steps_remaining} steps left! You MUST submit a diagnosis NOW or score 0. Use action_type=submit_diagnosis immediately!\n"
    elif steps_remaining <= 4:
        urgency = f"\n⚠️ WARNING: Only {steps_remaining} steps remaining. Submit diagnosis soon or you will timeout with score 0!\n"

    context = f"""Task: {task_id} (difficulty: {difficulty})
Step: {step} | Steps remaining: {steps_remaining}
Nodes inspected so far: {nodes_inspected}
Last action result: {last_result}

Pipeline nodes:
{chr(10).join(node_summary)}

Recent history (last 5 steps):
{chr(10).join(history[-5:]) if history else 'None yet'}

Last reward: {last_reward:.4f}
{urgency}
REMEMBER: You MUST use action_type=submit_diagnosis to get a real score. Inspecting without diagnosing = score of 0.
What is your next action? Respond with a JSON action object."""

    return context


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def get_model_action(
    client: OpenAI,
    step: int,
    last_observation: Dict[str, Any],
    last_reward: float,
    history: List[str],
) -> Dict[str, Any]:
    """Ask the LLM for the next action. Returns a parsed action dict."""
    user_prompt = build_user_prompt(step, last_observation, last_reward, history)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()

        # Extract JSON from response
        # Handle markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        action = json.loads(text)

        # Coerce old string-format diagnosis to new structured format
        if action.get("action_type") in ("diagnose", "submit_diagnosis"):
            diag = action.get("diagnosis")
            if isinstance(diag, str):
                # Parse "node_id1[,node_id2] | explanation" → structured dict
                parts = diag.split("|", 1)
                nodes_part = parts[0].strip()
                failure_type = parts[1].strip() if len(parts) > 1 else "unknown"
                node_ids = [n.strip() for n in nodes_part.replace(",", " ").split() if n.strip()]
                action["diagnosis"] = {
                    "root_cause_nodes": node_ids,
                    "failure_type": failure_type,
                }
            action["action_type"] = "submit_diagnosis"

        return action

    except json.JSONDecodeError:
        print(f"[DEBUG] Failed to parse JSON from model response: {text[:200]}", flush=True)
        return {"action_type": "list_nodes"}
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return {"action_type": "list_nodes"}


# ---------------------------------------------------------------------------
# Environment HTTP client
# ---------------------------------------------------------------------------


def env_reset(http_client: httpx.Client, task_id: str) -> Dict[str, Any]:
    resp = http_client.post("/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_step(http_client: httpx.Client, action: Dict[str, Any]) -> Dict[str, Any]:
    resp = http_client.post("/step", json=action, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Run one task
# ---------------------------------------------------------------------------


def run_task(
    llm_client: OpenAI,
    http_client: httpx.Client,
    task_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run the agent on a single task. Returns task results."""
    task_id = task_config["task_id"]
    max_steps = task_config["max_steps"]
    max_total_reward = task_config["max_total_reward"]

    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # Reset environment
        result = env_reset(http_client, task_id)
        last_reward = 0.0

        for step in range(1, max_steps + 1):
            done = result.get("done", False)
            if done:
                break

            # Get action from LLM
            action = get_model_action(
                llm_client, step, result, last_reward, history
            )

            # Validate action has required action_type
            if "action_type" not in action:
                action = {"action_type": "list_nodes"}

            # Execute action in environment
            error = None
            try:
                result = env_step(http_client, action)
                reward = result.get("reward", 0.0)
                done = result.get("done", False)
            except Exception as e:
                error = str(e)
                reward = 0.0
                done = False
                print(f"[DEBUG] Step failed: {e}", flush=True)

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            action_str = json.dumps(action)
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)
            history.append(f"Step {step}: {action_str} -> reward {reward:+.4f}")

            if done:
                break

        # Compute final score
        if rewards:
            # Score = final reward (the diagnosis reward), clamped to [0,1]
            # For non-diagnosed episodes, use mean reward
            final_reward = rewards[-1] if rewards else 0.0
            score = min(max(final_reward, 0.0), 1.0)
        else:
            score = 0.0

        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Task {task_id} failed: {e}", flush=True)
        error = str(e)

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task_id": task_id,
        "success": success,
        "steps": steps_taken,
        "score": score,
        "rewards": rewards,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable is required")

    print(f"[DEBUG] Starting inference | model={MODEL_NAME} | env={ENV_BASE_URL}", flush=True)

    # Initialize LLM client
    llm_client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Initialize HTTP client for environment
    http_client = httpx.Client(base_url=ENV_BASE_URL)

    # Health check
    try:
        resp = http_client.get("/health", timeout=10)
        resp.raise_for_status()
        print(f"[DEBUG] Environment health check passed: {resp.json()}", flush=True)
    except Exception as e:
        print(f"[DEBUG] Environment health check failed: {e}", flush=True)
        print(f"[DEBUG] Make sure the environment is running at {ENV_BASE_URL}", flush=True)
        sys.exit(1)

    # Run all tasks
    all_results = []
    start_time = time.time()

    for task_config in TASKS:
        print(f"\n[DEBUG] Starting task: {task_config['task_id']}", flush=True)
        result = run_task(llm_client, http_client, task_config)
        all_results.append(result)

        elapsed = time.time() - start_time
        print(f"[DEBUG] Task complete in {elapsed:.1f}s | score={result['score']:.4f}", flush=True)

        # Safety: abort if total time exceeds 18 minutes (leave 2 min buffer)
        if elapsed > 18 * 60:
            print("[DEBUG] Time limit approaching — stopping early", flush=True)
            break

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("FINAL RESULTS", flush=True)
    print("=" * 60, flush=True)
    for r in all_results:
        status = "PASS" if r["success"] else "FAIL"
        print(f"  [{status}] {r['task_id']}: score={r['score']:.4f} steps={r['steps']}", flush=True)

    mean_score = sum(r["score"] for r in all_results) / len(all_results) if all_results else 0.0
    print(f"\nMean score across all tasks: {mean_score:.4f}", flush=True)
    print("=" * 60, flush=True)

    http_client.close()


if __name__ == "__main__":
    main()
