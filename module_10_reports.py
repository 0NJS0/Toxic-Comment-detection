import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from evaluation.reports import generate_all


def main():
    print("=" * 60)
    print("MODULE 10: Reports & Plots")
    print("=" * 60)

    generate_all("results/reports")

    print(f"\n{'='*60}")
    print("MODULE 10 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
