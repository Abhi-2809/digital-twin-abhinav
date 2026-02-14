"""Run generation pipeline on test_queries.json for each model and save results."""

import json
import time
from pathlib import Path

from src.rag.router import route_query
from src.rag.retrievers import retrieve_documents
from src.rag.generator import format_documents_as_list, _build_prompt
from src.rag.utils import validate_api_key, get_llm

TEST_QUERIES_PATH = Path(__file__).resolve().parent / "test_queries.json"
test_models = [
    "gpt-4o-mini",        # Current working (temp=0.1)
    "gpt-4.1-nano",       # Nano + full temp control
    "gpt-4.1-mini",      # Best small model performance
]

# Price per 1M tokens (input, output)
PRICING = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4.1-mini": (0.40, 1.60),
}


def _model_to_filename(model: str) -> str:
    return model.replace(" ", "_") + "_testing.json"


def _get_usage(response) -> tuple[int, int]:
    meta = getattr(response, "response_metadata", None) or {}
    usage = meta.get("usage") or meta.get("token_usage") or meta
    prompt_tokens = int(
        usage.get("input_tokens")
        or usage.get("prompt_tokens")
        or 0
    )
    completion_tokens = int(
        usage.get("output_tokens")
        or usage.get("completion_tokens")
        or 0
    )
    return prompt_tokens, completion_tokens


def _compute_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    prices = PRICING.get(model, (0.0, 0.0))
    return (prompt_tokens / 1e6) * prices[0] + (completion_tokens / 1e6) * prices[1]


def run():
    validate_api_key()
    with open(TEST_QUERIES_PATH) as f:
        test_queries = json.load(f)

    for model in test_models:
        results = []
        for item in test_queries:
            user_input = item["user_input"]
            reference = item["reference"]
            route_decision = route_query(user_input)
            documents = retrieve_documents(
                query=user_input,
                collection=route_decision["collection"],
                strategy="semantic",
                k=route_decision["k"],
            )
            retrieved_contexts = format_documents_as_list(documents)
            llm = get_llm(model=model, temperature=0.1)
            formatted_prompt = _build_prompt(
                documents, user_input, route_decision["collection"]
            )
            t0 = time.perf_counter()
            response = llm.invoke(formatted_prompt)
            t1 = time.perf_counter()
            time_taken = round(t1 - t0, 4)
            prompt_tokens, completion_tokens = _get_usage(response)
            cost = round(_compute_cost(model, prompt_tokens, completion_tokens), 10)
            content = response.content if hasattr(response, "content") else str(response)
            results.append({
                "user_input": user_input,
                "response": content,
                "reference": reference,
                "retrieved_contexts": retrieved_contexts,
                "time_taken": time_taken,
                "cost": cost,
            })
        out_path = Path(__file__).resolve().parent / _model_to_filename(model)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    run()
