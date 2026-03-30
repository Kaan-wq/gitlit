from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.services.embeddings import load_model
from app.services.llm import load_llm
from fastapi.middleware.cors import CORSMiddleware
from app.routes import router
import os


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()
    app.state.llm, app.state.tokenizer = load_llm()
    yield
    del app.state.model
    del app.state.llm
    del app.state.tokenizer

app = FastAPI(
    title="GitLit",
    description="A platform to chat with your git repositories using natural language queries.",
    version="0.1.0",
    lifespan=lifespan,
)

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
