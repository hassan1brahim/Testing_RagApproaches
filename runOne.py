import subprocess
from datetime import datetime

QUESTIONS = [
    "What accounts and tools does a new Office of Innovation team member need to set up in their first week, and what security requirements apply to each?",
    "Which onboarding items differ for temporary employees versus full-time employees, and who should be contacted if issues arise?",
    "Summarize the full onboarding process from day one through the first few months, including required trainings and onboarding sessions."
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
