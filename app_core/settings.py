import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AppDefaults:
    chunk_size: int = 800
    chunk_overlap: int = 120
    retrieval_k: int = 4
    llm_provider: str = "ollama"
    model_name: str = "llama3.2:3b"
    temperature: float = 0.0
    ollama_base_url: str = "http://localhost:11434"


DEFAULTS = AppDefaults()


def get_openai_api_key() -> str:
    return os.getenv("OPENAI_API_KEY", "")
