from fastapi import Request
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_model(request: Request) -> SentenceTransformer:
    return request.app.state.model

def get_llm(request: Request) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    return request.app.state.llm, request.app.state.tokenizer
