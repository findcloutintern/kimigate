import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from routes import router
from provider import cleanup_provider

LOG_FILE = "server.log"

if not logging.root.handlers:
    open(LOG_FILE, "w", encoding="utf-8").close()
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8", mode="a")],
    )

logger = logging.getLogger(__name__)

logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("starting kimigate proxy...")
    yield
    await cleanup_provider()
    logger.info("shutting down...")


def create_app() -> FastAPI:
    app = FastAPI(
        title="kimigate",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.include_router(router)

    @app.exception_handler(Exception)
    async def error_handler(request: Request, exc: Exception):
        logger.error(f"error: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": str(exc),
                },
            },
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="debug")
