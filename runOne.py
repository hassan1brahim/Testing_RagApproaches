import subprocess
from datetime import datetime

# Write your questions for testing
QUESTIONS = [ 
]

OUTPUT_FILE = "query_results.txt"

def run_query(question):
    result = subprocess.run(
        ["python", "query.py", question],
        capture_output=True,
        text=True
    )
    return result.stdout.strip()

def main():
    with open(OUTPUT_FILE, "a") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Run at: {datetime.now()}\n")
        f.write("=" * 80 + "\n\n")

        for q in QUESTIONS:
            f.write(f"Q: {q}\n")
            f.write("-" * 80 + "\n")

            answer = run_query(q)
            f.write(answer + "\n\n")

    print("Done. Results appended to", OUTPUT_FILE)

if __name__ == "__main__":
    main()
