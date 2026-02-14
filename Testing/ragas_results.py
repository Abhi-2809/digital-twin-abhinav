"""Compute RAGAS metrics on testing JSON outputs and print a 3x5 summary table.

Output (table + progress) is printed to stdout and appended to OUTPUT_TXT.
To run in background and close laptop, use screen:

  cd /path/to/digital-twin-viven
  screen -S ragas
  PYTHONPATH=. python Testing/ragas_results.py
  # Detach: Ctrl+A then D
  # Reattach later: screen -r ragas
  # View output anytime: cat Testing/ragas_results_output.txt
"""

import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import NoiseSensitivity, AnswerRelevancy, Faithfulness

TESTING_DIR = Path(__file__).resolve().parent
OUTPUT_TXT = TESTING_DIR / "ragas_results_output.txt"
OUTPUT_JSON = TESTING_DIR / "ragas_results.json"
TEST_FILES = [
    "gpt-4o-mini_testing.json",
    "gpt-4.1-nano_testing.json",
    "gpt-4.1-mini_testing.json",
]
SLEEP_EVERY_N_CALLS = 5
SLEEP_SEC = 2

_log_file_ref = [None]


def _log(msg: str) -> None:
    print(msg, flush=True)
    if _log_file_ref[0] is not None:
        _log_file_ref[0].write(msg + "\n")
        _log_file_ref[0].flush()


async def run():
    client = AsyncOpenAI()
    llm = llm_factory("gpt-4o-mini", client=client)
    embeddings = embedding_factory("openai", model="text-embedding-3-small", client=client)

    noise_scorer = NoiseSensitivity(llm=llm)
    relevancy_scorer = AnswerRelevancy(llm=llm, embeddings=embeddings)
    faithfulness_scorer = Faithfulness(llm=llm)

    rows = []
    call_count = 0
    for file_idx, filename in enumerate(TEST_FILES):
        path = TESTING_DIR / filename
        if not path.exists():
            _log(f"Skipping {filename} (not found)")
            continue
        model_name = filename.replace("_testing.json", "")
        with open(path) as f:
            data = json.load(f)

        n = len(data)
        _log(f"[{file_idx + 1}/{len(TEST_FILES)}] Model: {model_name} ({n} queries)")

        noise_vals = []
        relevancy_vals = []
        faithfulness_vals = []
        time_vals = []
        cost_vals = []

        for i, item in enumerate(data):
            _log(f"  Query {i + 1}/{n}: {item['user_input'][:50]}...")
            user_input = item["user_input"]
            response = item["response"]
            reference = item["reference"]
            retrieved_contexts = item["retrieved_contexts"]
            time_vals.append(item["time_taken"])
            cost_vals.append(item["cost"])

            ns = await noise_scorer.ascore(
                user_input=user_input,
                response=response,
                reference=reference,
                retrieved_contexts=retrieved_contexts,
            )
            noise_vals.append(ns.value)
            call_count += 1
            if call_count % SLEEP_EVERY_N_CALLS == 0:
                await asyncio.sleep(SLEEP_SEC)

            ar = await relevancy_scorer.ascore(
                user_input=user_input,
                response=response,
            )
            relevancy_vals.append(ar.value)
            call_count += 1
            if call_count % SLEEP_EVERY_N_CALLS == 0:
                await asyncio.sleep(SLEEP_SEC)

            fth = await faithfulness_scorer.ascore(
                user_input=user_input,
                response=response,
                retrieved_contexts=retrieved_contexts,
            )
            faithfulness_vals.append(fth.value)
            call_count += 1
            if call_count % SLEEP_EVERY_N_CALLS == 0:
                await asyncio.sleep(SLEEP_SEC)

        rows.append({
            "model": model_name,
            "noise_sensitivity": sum(noise_vals) / n,
            "answer_relevancy": sum(relevancy_vals) / n,
            "faithfulness": sum(faithfulness_vals) / n,
            "avg_time_taken": sum(time_vals) / n,
            "avg_cost": sum(cost_vals) / n,
        })

    with open(OUTPUT_JSON, "w") as f:
        json.dump(rows, f, indent=2)
    _log(f"Results saved to {OUTPUT_JSON}")

    col_w = 18
    headers = ("Model", "Noise Sensitivity", "Response Relevancy", "Faithfulness", "Avg time_taken", "Avg cost")
    fmt = ("| {{:<{}}} " * 6 + "|").format(*[col_w] * 6)
    sep = "+" + "-" * (col_w + 2) * 6 + "+"
    _log("")
    _log(sep)
    _log(fmt.format(*headers))
    _log(sep)
    for r in rows:
        _log(fmt.format(
            r["model"],
            round(r["noise_sensitivity"], 4),
            round(r["answer_relevancy"], 4),
            round(r["faithfulness"], 4),
            round(r["avg_time_taken"], 4),
            round(r["avg_cost"], 6),
        ))
    _log(sep)


if __name__ == "__main__":
    with open(OUTPUT_TXT, "w") as f:
        _log_file_ref[0] = f
        _log(f"ragas_results.py started (output file: {OUTPUT_TXT})")
        _log("")
        try:
            asyncio.run(run())
        finally:
            _log("")
            _log("Done.")
        _log_file_ref[0] = None
