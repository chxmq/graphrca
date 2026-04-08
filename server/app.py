"""Entry point for openenv multi-mode deployment."""

import os
import uvicorn
from app import app


def main() -> None:
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
