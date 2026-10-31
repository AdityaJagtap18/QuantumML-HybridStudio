from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import architectures, datasets, experiments
from app.core.config import get_settings
from app.core.database import init_db

settings = get_settings()
init_db()

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets.router, prefix=settings.api_prefix)
app.include_router(architectures.router, prefix=settings.api_prefix)
app.include_router(experiments.router, prefix=settings.api_prefix)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
