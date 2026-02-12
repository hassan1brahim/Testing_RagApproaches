# =============================================================================
# run_experiment.py — Compare RAG approaches
# =============================================================================
#
# Runs the same set of questions through all three query approaches:
# 1. Baseline (query.py) — simple context + question
# 2. Chain of Thought (query_cot.py) — adds reasoning steps
# 3. CoT + Citations (query_cot_cite.py) — reasoning + source tracking
#
# Results are saved to a text file for comparison.
#
# Usage:
#   python run_experiment.py
#
# =============================================================================

from datetime import datetime

from query import query_rag as query_baseline
from query_cot import query_rag as query_cot
from query_cot_cite import query_rag as query_cot_cite


# =============================================================================
# TEST QUESTIONS
# =============================================================================
# Add your test questions here. Use questions that will help you evaluate
# the differences between approaches.
# =============================================================================

# Write your questions for testing
QUESTIONS = [ 
]

OUTPUT_FILE = "experiment_results.txt"


def run_experiment():
    """
    Runs all questions through each approach and saves results.
    """

    print(f"Running experiment with {len(QUESTIONS)} questions...")
    print(f"Testing 3 approaches: Baseline, CoT, CoT+Citations\n")

    with open(OUTPUT_FILE, "w") as f:
        # Header
        f.write("RAG Experiment Results\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Questions: {len(QUESTIONS)}\n")
        f.write("=" * 70 + "\n\n")

        for i, question in enumerate(QUESTIONS, 1):
            print(f"[{i}/{len(QUESTIONS)}] {question}")

            f.write(f"QUESTION {i}: {question}\n")
            f.write("-" * 70 + "\n\n")

            # --- Baseline ---
            print("  - Running baseline...")
            f.write("[BASELINE]\n")
            try:
                result = query_baseline(question)
                f.write(f"{result}\n\n")
            except Exception as e:
                f.write(f"ERROR: {e}\n\n")

            # --- Chain of Thought ---
            print("  - Running CoT...")
            f.write("[CHAIN OF THOUGHT]\n")
            try:
                result = query_cot(question)
                f.write(f"{result}\n\n")
            except Exception as e:
                f.write(f"ERROR: {e}\n\n")

            # --- CoT + Citations ---
            print("  - Running CoT + Citations...")
            f.write("[COT + CITATIONS]\n")
            try:
                result = query_cot_cite(question)
                f.write(f"{result['answer']}\n\n")
                f.write("Sources:\n")
                for src in result['sources']:
                    f.write(f"  - Source {src['source_num']}: {src['filename']}, ")
                    f.write(f"Page {src['page']}, Chunk: {src['chunk_id']}\n")
                f.write("\n")
            except Exception as e:
                f.write(f"ERROR: {e}\n\n")

            f.write("=" * 70 + "\n\n")

    print(f"\nExperiment complete. Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_experiment()
