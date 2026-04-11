"""Graph-RCA Pipeline Diagnoser — OpenEnv FastAPI server."""

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from environment.env import GraphRCAEnv
from environment.models import ActionType, DiagnosisInput, PipelineAction

app = FastAPI(
    title="Graph-RCA Pipeline Diagnoser",
    description=(
        "OpenEnv environment for diagnosing root causes in failing data pipelines. "
        "Built on 200 real annotated production incidents using graph-based analysis."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance (stateful per-server; each client should use reset())
_env = GraphRCAEnv()


class ResetRequest(BaseModel):
    task_id: Optional[str] = "single_point_failure"


class StepRequest(BaseModel):
    action_type: str
    target_node: Optional[str] = None
    diagnosis: Optional[DiagnosisInput] = None


@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "name": "graph-rca-pipeline-diagnoser",
        "version": "1.0.0",
        "description": "OpenEnv environment for pipeline root cause analysis",
        "tasks": GraphRCAEnv.AVAILABLE_TASKS,
        "endpoints": {
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state",
            "health": "GET /health",
            "tasks": "GET /tasks",
        },
    }


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "healthy", "environment": "graph-rca-pipeline-diagnoser"}


@app.get("/tasks")
async def list_tasks() -> Dict[str, Any]:
    return {
        "tasks": [
            {
                "task_id": "single_point_failure",
                "difficulty": "easy",
                "description": "Single node failure with explicit error in logs",
                "max_steps": 10,
            },
            {
                "task_id": "cascading_failure",
                "difficulty": "medium",
                "description": "Cascading failure — root cause is upstream of visible errors",
                "max_steps": 15,
            },
            {
                "task_id": "simultaneous_failures",
                "difficulty": "hard",
                "description": "Two simultaneous failures — one active, one monitoring blind spot",
                "max_steps": 20,
            },
        ]
    }


@app.post("/reset")
async def reset(request: Optional[ResetRequest] = None) -> Dict[str, Any]:
    """Reset the environment with a task. Returns initial observation."""
    try:
        task_id = request.task_id if request else "single_point_failure"
        result = _env.reset(task_id=task_id)
        return result.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.post("/step")
async def step(request: StepRequest) -> Dict[str, Any]:
    """Execute one action. Returns observation, reward, done, info."""
    try:
        # Validate action type
        try:
            action_type = ActionType(request.action_type)
        except ValueError:
            valid = [a.value for a in ActionType]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid action_type '{request.action_type}'. Valid: {valid}",
            )

        action = PipelineAction(
            action_type=action_type,
            target_node=request.target_node,
            diagnosis=request.diagnosis,
        )
        result = _env.step(action)
        return result.to_dict()

    except HTTPException:
        raise
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")


@app.get("/state")
async def state() -> Dict[str, Any]:
    """Return current environment state."""
    return _env.state()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
