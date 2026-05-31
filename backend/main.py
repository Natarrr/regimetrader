from fastapi import FastAPI

app = FastAPI(
    title="Trading Dashboard Backend",
    version="0.1.0",
    description="Minimal FastAPI entrypoint for the Trading Dashboard backend.",
)


@app.get("/", tags=["health"])
async def root() -> dict[str, str]:
    return {"status": "ok", "service": "trading-backend"}


@app.get("/healthz", tags=["health"])
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
