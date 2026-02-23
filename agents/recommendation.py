import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from state.schema import AssistantState

_llm = ChatOllama(model="llama2", temperature=0)

_COMPARISON_SYSTEM_PROMPT = """You are a structured decision-support engine.

When a student compares two or more career paths, you do NOT choose one.
You design a 2-week structured exploration experiment.

STRICT RULES:
- Return ONLY valid JSON.
- No explanations.
- No markdown.
- No emojis.
- No prose outside JSON.
- exploration_plan must contain exactly two items (Week 1 and Week 2).

Return a JSON object with exactly these keys:
  - strategy              : string (must be "exploration_experiment")
  - exploration_plan      : list of 2 dicts with keys {week, focus, task}
  - decision_criteria     : list of 3 reflective evaluation questions
  - final_suggestion      : short decision guidance sentence
  - resources             : list of 2–4 learning resources
"""

_DIRECT_PATH_SYSTEM_PROMPT = """You are a career recommendation engine for a Student Planning Assistant.

You receive a structured interpretation of a student's goal and constraints.
Return a JSON object with exactly these keys:
  - primary_path        : string — the most suitable career direction given the context
  - secondary_path      : string — a viable alternative
  - reasoning           : string — 2–3 sentences explaining why primary_path fits best
  - immediate_next_steps: list of strings — 3 concrete actions the student can take this week
  - resources           : list of strings — 2–3 specific learning resources or links

Return ONLY valid JSON. No prose. No markdown fences.
"""


def _generate_comparison_recommendation(goal: str, constraints: list, missing: list) -> dict:
    """
    Generate a structured exploration plan for comparison decisions.
    Includes schema validation and deterministic fallback.
    """

    context_block = (
        f"Student goal: {goal}\n"
        f"Explicit constraints: {', '.join(constraints) if constraints else 'none stated'}\n"
        f"Missing information: {', '.join(missing) if missing else 'none'}"
    )

    messages = [
        SystemMessage(content=_COMPARISON_SYSTEM_PROMPT),
        HumanMessage(content=context_block),
    ]

    response = _llm.invoke(messages)

    try:
        recommendations = json.loads(response.content)
    except json.JSONDecodeError:
        return _comparison_fallback(goal)

    # ---- Schema validation ----
    required_keys = {
        "strategy",
        "exploration_plan",
        "decision_criteria",
        "final_suggestion",
        "resources",
    }

    if not isinstance(recommendations, dict):
        return _comparison_fallback(goal)

    if not required_keys.issubset(recommendations.keys()):
        return _comparison_fallback(goal)

    if not recommendations.get("exploration_plan"):
        return _comparison_fallback(goal)

    return recommendations


def _comparison_fallback(goal: str) -> dict:
    """
    Deterministic fallback if LLM fails to return valid structured output.
    Ensures system robustness and avoids empty plans.
    """

    return {
        "strategy": "exploration_experiment",
        "exploration_plan": [
            {
                "week": 1,
                "focus": "Backend Development",
                "task": "Build a small REST API using FastAPI with authentication and database integration.",
            },
            {
                "week": 2,
                "focus": "Machine Learning",
                "task": "Build a small end-to-end ML classification project and expose it via a simple API.",
            },
        ],
        "decision_criteria": [
            "Which work felt more engaging?",
            "Which learning curve felt sustainable?",
            "Which type of problem-solving excited you more?",
        ],
        "final_suggestion": "Choose the path that maintains your long-term curiosity after hands-on exploration.",
        "resources": [
            "FastAPI official documentation",
            "Scikit-learn documentation",
            "Andrew Ng Machine Learning Course",
        ],
    }


def _generate_direct_recommendation(goal: str, constraints: list, missing: list) -> dict:

    context_block = (
        f"Student goal: {goal}\n"
        f"Constraints / preferences: {', '.join(constraints) if constraints else 'none stated'}\n"
        f"Note — information not available: {', '.join(missing) if missing else 'none'}"
    )

    messages = [
        SystemMessage(content=_DIRECT_PATH_SYSTEM_PROMPT),
        HumanMessage(content=context_block),
    ]

    response = _llm.invoke(messages)

    try:
        recommendations = json.loads(response.content)
    except json.JSONDecodeError:
        recommendations = {
            "primary_path": "Unable to parse recommendations.",
            "secondary_path": "N/A",
            "reasoning": "LLM returned unexpected format.",
            "immediate_next_steps": [],
            "resources": [],
        }

    return recommendations


def run_recommendation(state: AssistantState) -> AssistantState:
    """
    Node function: Recommendation Agent.

    Reads:  interpreted_goal, extracted_constraints, missing_information, decision_type
    Writes: recommendations
    """
    goal = state.interpreted_goal or state.user_input
    constraints = state.extracted_constraints or []
    missing = state.missing_information or []
    decision_type = state.decision_type or "direct_path"

    print(f"\n[RecommendationAgent] Generating recommendations for goal: '{goal}'")
    print(f"[RecommendationAgent] Decision type: {decision_type}")

    if decision_type == "comparison":
        recommendations = _generate_comparison_recommendation(goal, constraints, missing)
        print(f"[RecommendationAgent] Strategy: exploration_experiment")
    else:
        recommendations = _generate_direct_recommendation(goal, constraints, missing)
        print(f"[RecommendationAgent] Primary path: {recommendations.get('primary_path')}")

    return state.model_copy(update={"recommendations": recommendations})
