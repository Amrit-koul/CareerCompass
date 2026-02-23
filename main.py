"""
main.py

CLI entry point for the Student Planning Assistant.

Usage:
    python main.py
    python main.py --query "Should I focus on ML or backend development?"

The script:
1. Accepts a student query (from CLI arg or interactive prompt)
2. Initializes AssistantState with the query
3. Runs the compiled LangGraph workflow
4. Prints the final response and intermediate state for inspection
"""

import argparse
import json
import os

from state.schema import AssistantState
from graph.workflow import build_graph


def run_assistant(query: str) -> None:
    """
    Execute the full multi-agent workflow for a given student query.
    """
    print("\n" + "=" * 60)
    print("  Student Planning Assistant — Multi-Agent System")
    print("=" * 60)
    print(f"\nStudent query: \"{query}\"\n")

    # Build the graph
    app = build_graph()

    # Initialize state — only user_input is required at the start
    initial_state = AssistantState(user_input=query)

    # LangGraph expects a plain dict
    result_dict = app.invoke(initial_state.model_dump())

    # Deserialize final state for clean access
    final_state = AssistantState.model_validate(result_dict)

    # --- Print results ---
    print("\n" + "=" * 60)
    print("  FINAL RESPONSE")
    print("=" * 60)
    print(final_state.final_response)

    print("\n" + "=" * 60)
    print("  INTERMEDIATE STATE SUMMARY")
    print("=" * 60)
    print(f"Confidence score     : {final_state.confidence_score:.2f}")
    print(f"Interpreted goal     : {final_state.interpreted_goal}")
    print(f"Constraints          : {final_state.extracted_constraints}")

    if final_state.recommendations:
        print("\nRecommendations (structured):")
        print(json.dumps(final_state.recommendations, indent=2))

    print("\n" + "=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Student Planning Assistant — A2A LangGraph Demo"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Student query string. If omitted, you will be prompted interactively.",
    )
    args = parser.parse_args()

    if args.query:
        query = args.query
    else:
        query = input("Enter your planning question: ").strip()
        if not query:
            query = "I'm a second-year CS student unsure whether to focus on ML or backend development. What should I do?"
            print(f"Using default query: \"{query}\"")

    run_assistant(query)


if __name__ == "__main__":
    main()
