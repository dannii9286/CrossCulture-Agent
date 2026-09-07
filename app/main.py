from fastapi import FastAPI

from app.api.v1.routes import router as api_v1_router


def create_app() -> FastAPI:
    application = FastAPI(
        title="CrossCulture-Agent",
        description="Cross-cultural naming API (phase 1 mock implementation).",
        version="0.1.0",
    )
    application.include_router(api_v1_router, prefix="/api/v1")
    return application


app = create_app()
