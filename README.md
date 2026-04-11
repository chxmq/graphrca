---
title: Graph RCA Pipeline Diagnoser
emoji: 🔍
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
tags:
  - openenv
  - data-engineering
  - root-cause-analysis
---

# Graph-RCA Pipeline Diagnoser

An OpenEnv reinforcement learning environment where an AI agent diagnoses root causes in failing production data pipelines by traversing a DAG, inspecting logs, and identifying what broke and why.

**Built on 200 real annotated production incidents** from Cloudflare, AWS, GitHub, Google, Allegro, Railway, and more.

---

## Why This Environment

RCA is what SREs actually do under pressure — trace a live failure through a system they don't fully understand, fast. This environment models that directly:

- Built on **real incident data** (not toy scenarios)
- Requires **causal reasoning** across a graph — not just pattern matching
- Scales from single-node failures to simultaneous multi-node failures

---

## Environment Description

The agent is given a **pipeline DAG** (Directed Acyclic Graph) where each node represents a service, database, network component, or monitoring system. Some nodes are failing; the agent must:

1. Explore the DAG by inspecting node logs
2. Trace causal chains backward through the graph
3. Identify the root cause node(s)
4. Submit a diagnosis

The reward signal is **dense** — the agent gets signal at every step, not just at the end.

---

## Action Space

| Action | Description | Parameters |
|--------|-------------|------------|
| `list_nodes` | Show all pipeline nodes with status | none |
| `inspect_node` | Read logs of a specific node | `target_node` (required) |
| `traverse_edge` | Show edges to/from a node | `target_node` (required) |
| `get_metrics` | Get pipeline performance metrics | none |
| `submit_diagnosis` | Submit root cause diagnosis (ends episode) | `diagnosis` (required) |

**Action format** (JSON):
```json
{"action_type": "inspect_node", "target_node": "node_cluster_manager"}
{"action_type": "submit_diagnosis", "diagnosis": {"root_cause_nodes": ["node_waf_deploy"], "failure_type": "catastrophic_regex_backtracking"}}
```

---

## Observation Space

Each observation contains:

| Field | Type | Description |
|-------|------|-------------|
| `task_id` | string | Current task identifier |
| `difficulty` | string | easy / medium / hard |
| `nodes` | array | All nodes with status (HEALTHY/DEGRADED/FAILED) and error flags |
| `edges` | array | DAG edges as `{from, to}` pairs |
| `inspected_node` | string | Node being inspected (after inspect_node) |
| `inspected_logs` | array | Log entries: timestamp, level, message |
| `metrics` | object | Pipeline metrics (after get_metrics) |
| `current_step` | int | Current step number |
| `max_steps` | int | Episode step limit |
| `steps_remaining` | int | Steps left |
| `nodes_inspected` | array | All nodes inspected so far |
| `last_action_result` | string | Feedback from last action |

---

## Reward Function

```
final_score = (F1_score x 0.80) + (efficiency_bonus x 0.20)
```

- **F1 score**: Precision x Recall of predicted vs true root cause nodes
- **Efficiency bonus**: Up to 0.20 for solving in fewer steps
- **Exploration reward**: Small dense signal for inspecting root cause nodes
- **Step penalty**: Tiny negative reward to discourage random wandering

Scores are always in **(0.001, 0.999)**.

---

## Tasks

Baseline scores measured with `gpt-4o-mini`, `temperature=0`.

| Task | Difficulty | Source | Scenario | Root Causes | Max Steps | Score (gpt-4o-mini) |
|------|------------|--------|----------|-------------|-----------|---------------------|
| `single_point_failure` | Easy | Allegro (2018) | CPU/RAM misconfiguration blocks deployments | 1 node — visible in CRITICAL logs | 10 | 0.84 |
| `cascading_failure` | Medium | Cloudflare WAF (2019) | Regex backtracking causes CPU exhaustion across edge routers | 1 node — upstream of visible failures | 15 | 0.93 |
| `simultaneous_failures` | Hard | Cloudflare BGP (2020) | BGP config typo + alert suppression fire simultaneously | 2 nodes — one active failure, one monitoring blind spot | 20 | 0.35 |
| **Mean** | — | — | — | — | — | **0.71** |

---

## Setup & Usage

### 1. Configure environment variables

```bash
cp .env.example .env
# Fill in HF_TOKEN and optionally API_BASE_URL / MODEL_NAME
```

### 2. Run locally

```bash
uv sync
uv run uvicorn app:app --host 0.0.0.0 --port 7860
```

### 3. Run baseline inference

```bash
# Canonical submission command
API_BASE_URL=https://api.openai.com/v1 \
MODEL_NAME=gpt-4o-mini \
HF_TOKEN=your-key-here \
ENV_BASE_URL=http://localhost:7860 \
uv run python inference.py
```

### Docker

```bash
docker build -t graph-rca-env .
docker run -p 7860:7860 graph-rca-env
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id": "single_point_failure"}'
```

### API Examples

```bash
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id": "single_point_failure"}'
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d '{"action_type": "inspect_node", "target_node": "node_cluster_manager"}'
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d '{"action_type": "submit_diagnosis", "diagnosis": {"root_cause_nodes": ["node_cluster_manager"], "failure_type": "resource_misconfiguration"}}'
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `API_BASE_URL` | Yes | LLM API endpoint |
| `MODEL_NAME` | Yes | Model identifier |
| `HF_TOKEN` | Yes | HuggingFace / API key |
| `ENV_BASE_URL` | No | Environment URL (default: `http://localhost:7860`) |

---

## Project Structure

```
graph-rca/
├── environment/
│   ├── env.py          # Main OpenEnv class (reset/step/state)
│   ├── models.py       # Pydantic models (Observation/Action/Reward)
│   ├── grader.py       # Deterministic F1-based scoring
│   └── tasks/
│       ├── task_easy.json    # single_point_failure
│       ├── task_medium.json  # cascading_failure
│       └── task_hard.json    # simultaneous_failures
├── app.py              # FastAPI server with OpenEnv HTTP endpoints
├── inference.py        # Baseline agent (OpenAI API client)
├── openenv.yaml        # OpenEnv spec metadata
├── Dockerfile          # Container for HuggingFace Spaces
└── requirements.txt
```
