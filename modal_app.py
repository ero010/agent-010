"""Modal deployment for Agent 010 (free tier, no card needed).
Run: modal deploy modal_app.py   (from the life-manager folder)
"""
import modal

vol = modal.Volume.from_name("agent010-data", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install("fastapi", "uvicorn", "python-multipart", "openai", "pydantic")
    .add_local_file("backend.py", "/app/backend.py")
    .add_local_file("index.html", "/app/index.html")
)

app = modal.App("agent-010")


@app.function(image=image, secrets=[modal.Secret.from_name("agent-010-secrets")],
              volumes={"/data": vol})
@modal.asgi_app()
def api():
    import sys
    sys.path.insert(0, "/app")
    from backend import app as fastapi_app
    return fastapi_app
