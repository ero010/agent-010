"""HuggingFace entrypoint (Gradio SDK spaces run `python app.py`).
Serves the 010 FastAPI app directly on HF's port 7860."""
import os
import uvicorn

from backend import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
