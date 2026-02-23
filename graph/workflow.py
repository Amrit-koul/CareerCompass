from langgraph.graph import StateGraph, END

from state.schema import AssistantState
from agents.domain_reasoning import run_domain_reasoning
from agents.recommendation import run_recommendation
from agents.coordinator import compile_response


def build_graph() -> StateGraph:
    """
    Constructs and compiles the LangGraph workflow.

    Returns a compiled graph ready to be invoked with an initial AssistantState.
    """

    graph = StateGraph(dict)


    def _wrap(fn):
        """Converts dict <-> AssistantState at node boundaries."""
        def wrapped(state_dict: dict) -> dict:
            state = AssistantState.model_validate(state_dict)
            updated = fn(state)
            return updated.model_dump()
        return wrapped

    graph.add_node("domain_reasoning",     _wrap(run_domain_reasoning))
    graph.add_node("recommendation",       _wrap(run_recommendation))
    graph.add_node("compile_response",     _wrap(compile_response))

    # ------------------------------------------------------------------
    # Define edges 
    # ------------------------------------------------------------------

    # Entry point → domain reasoning always runs first
    graph.set_entry_point("domain_reasoning")

    # Linear flow: domain_reasoning → recommendation → compile_response
    graph.add_edge("domain_reasoning", "recommendation")
    graph.add_edge("recommendation", "compile_response")

    # Terminal node
    graph.add_edge("compile_response", END)

    return graph.compile()
