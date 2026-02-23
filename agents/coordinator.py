"""
agents/coordinator.py

Coordinator Agent (Supervisor) — Single Responsibility:
    Assemble the final response from recommendation output.

This agent does NOT interpret user intent, produce recommendations, or route flow.
Its only job is to compile structured recommendations into a final student-facing response.

The actual graph orchestration is defined in graph/workflow.py.
This module contains the node functions the coordinator executes.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from state.schema import AssistantState

_llm = ChatOllama(model="llama2", temperature=0)


# ---------------------------------------------------------------------------
# Node: compile_response
#   Takes recommendations from state and produces the final student-facing reply.
# ---------------------------------------------------------------------------

def compile_response(state: AssistantState) -> AssistantState:
    """
    Assembles the final response from domain context and recommendations.
    This is the last node before the graph terminates.
    
    For comparison decisions: formats deterministic exploration plan (no LLM).
    For direct-path decisions: uses LLM to synthesize roadmap.
    """
    recs = state.recommendations or {}
    goal = state.interpreted_goal or state.user_input
    decision_type = state.decision_type or "direct_path"

    print("\n[Coordinator] Compiling final response...")
    print(f"[Coordinator] Decision type for compilation: {decision_type}")

    if decision_type == "comparison":
        # Comparison mode: deterministic, structured formatting
        plan = recs.get("exploration_plan", [])
        criteria = recs.get("decision_criteria", [])
        suggestion = recs.get("final_suggestion", "")

        # Build response without LLM for determinism
        final = (
            "This is a comparison decision. Instead of prematurely selecting one path, "
            "I recommend a structured two-week exploration plan:\n\n"
        )

        if len(plan) >= 2:
            final += (
                f"**Week 1 — {plan[0].get('focus', 'Option A')}**\n"
                f"- {plan[0].get('task', 'Explore this path')}\n\n"
                f"**Week 2 — {plan[1].get('focus', 'Option B')}**\n"
                f"- {plan[1].get('task', 'Explore this path')}\n\n"
            )

        final += "After completing both phases, reflect on:\n"
        for i, criterion in enumerate(criteria[:3], 1):
            final += f"- {criterion}\n"

        if suggestion:
            final += f"\n{suggestion}"

        response_text = final
    else:
        # Direct path mode: use LLM to synthesize
        prompt = (
            f"You are a friendly student career advisor.\n"
            f"The student's goal: {goal}\n"
            f"Structured recommendations:\n{recs}\n\n"
            "Write a clear, encouraging, 3–5 sentence response that summarizes "
            "the top recommendation and one concrete next step. "
            "Address the student directly."
        )

        response = _llm.invoke([
            SystemMessage(content="You are a helpful student planning assistant."),
            HumanMessage(content=prompt),
        ])
        response_text = response.content.strip()

    return state.model_copy(update={"final_response": response_text})
