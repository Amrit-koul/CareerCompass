
import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from state.schema import AssistantState

_llm = ChatOllama(model="llama2", temperature=0)

_SYSTEM_PROMPT = """You are a domain reasoning engine for a Student Career Planning Assistant.

Your job is to parse the student's query and return a JSON object with exactly these keys:
  - interpreted_goal   : string — what the student is trying to decide or achieve
  - extracted_constraints : list of strings — preferences, limits, or context clues mentioned
  - missing_information   : list of strings — what additional info would sharpen the analysis
  - confidence_score      : float between 0.0 and 1.0

Confidence scoring guide:
  - 0.9–1.0 : query is specific, constraints are clear, goal is unambiguous
  - 0.7–0.89: goal is reasonably clear but some context is missing
  - 0.5–0.69: goal is vague or multiple interpretations are equally plausible
  - below 0.5: query is too ambiguous to reason about usefully

Return ONLY valid JSON. No prose. No markdown fences.
"""

# ---------------------------------------------------------------------------
# Helper: Extract explicitly mentioned constraints from user input
# ---------------------------------------------------------------------------
def _extract_explicit_constraints(user_input: str) -> list:
    """
    Extract constraints explicitly mentioned in the user input.
    Only flags constraints the user directly stated, not inferred claims.
    
    Returns:
        list of constraint strings (may be empty if none found)
    """
    explicit_keywords = {
        "time": ["limited time", "short on time", "urgent", "deadline", "soon"],
        "budget": ["limited budget", "can't afford", "cheap", "free"],
        "experience": ["beginner", "just starting", "no experience", "new to"],
        "priority": ["need a job", "job priority", "employment", "urgent"],
        "interest": ["love", "passion", "interested in", "enjoy"],
        "skill_gap": ["don't know", "unfamiliar", "never learned"],
    }
    
    user_lower = user_input.lower()
    found_constraints = []
    
    for category, keywords in explicit_keywords.items():
        if any(kw in user_lower for kw in keywords):
            found_constraints.append(category)
    
    return found_constraints


# ---------------------------------------------------------------------------
# Helper: Detect if input has background context
# ---------------------------------------------------------------------------
def _has_background_context(user_input: str) -> bool:
    """
    Check if user input contains background context (year, experience, skills).
    
    Returns:
        True if context is present, False otherwise
    """
    context_keywords = [
        # Year/level indicators
        "first year", "second year", "third year", "fourth year",
        "freshman", "sophomore", "junior", "senior",
        "year 1", "year 2", "year 3", "year 4",
        # Experience indicators
        "experience", "worked with", "built with", "used", "familiar with", "studied",
        # Skill mentions
        "python", "javascript", "java", "c++", "react", "django", "nodejs",
        "sql", "databases", "api", "aws", "cloud",
        # Current situation
        "currently", "doing", "working on", "taking", "course"
    ]
    
    user_lower = user_input.lower()
    return any(kw in user_lower for kw in context_keywords)


# ---------------------------------------------------------------------------
# Helper: Adjust confidence for comparison queries without context
# ---------------------------------------------------------------------------
def _adjust_confidence_for_comparison(
    confidence: float,
    decision_type: str,
    user_input: str
) -> float:
    """
    Realistically adjust confidence score for comparison queries.
    
    If it's a comparison without background context, lower confidence.
    This reflects that we don't have enough info to make a strong interpretation.
    
    Args:
        confidence: Original confidence from LLM
        decision_type: "comparison" or "direct_path"
        user_input: Original user input
    
    Returns:
        Adjusted confidence score
    """
    if decision_type != "comparison":
        return confidence
    
    has_context = _has_background_context(user_input)
    
    if not has_context:
        return min(confidence, 0.67)
    
    return confidence


def run_domain_reasoning(state: AssistantState) -> AssistantState:
    """
    Node function: Domain Reasoning Agent.

    Reads:  state.user_input
    Writes: interpreted_goal, extracted_constraints, missing_information, confidence_score, decision_type
    """
    active_input = state.user_input

    print(f"\n[DomainReasoningAgent] Analyzing input: '{active_input}'")

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=active_input),
    ]

    response = _llm.invoke(messages)

    try:
        parsed = json.loads(response.content)
    except json.JSONDecodeError:
        # Fallback: assign low confidence so the coordinator requests clarification
        parsed = {
            "interpreted_goal": "Unable to parse student goal.",
            "extracted_constraints": [],
            "missing_information": ["Please restate your question clearly."],
            "confidence_score": 0.3,
        }

    interpreted_goal = parsed.get("interpreted_goal", "").lower()
    comparison_keywords = [" or ", "vs", "which", "better", "compare"]
    
    if any(k in interpreted_goal for k in comparison_keywords):
        decision_type = "comparison"
    else:
        decision_type = "direct_path"

    
    explicit_constraints = _extract_explicit_constraints(active_input)
    
    print(f"[DomainReasoningAgent] Decision type: {decision_type}")
    print(f"[DomainReasoningAgent] Explicitly mentioned constraints: {explicit_constraints if explicit_constraints else 'none'}")
    
    original_confidence = float(parsed.get("confidence_score", 0.5))
    adjusted_confidence = _adjust_confidence_for_comparison(
        original_confidence,
        decision_type,
        active_input
    )
    
    if adjusted_confidence != original_confidence:
        print(f"[DomainReasoningAgent] Confidence adjusted: {original_confidence:.2f} → {adjusted_confidence:.2f}")
    
    print(f"[DomainReasoningAgent] Final confidence: {adjusted_confidence:.2f}")

    return state.model_copy(update={
        "interpreted_goal": parsed.get("interpreted_goal"),
        "extracted_constraints": explicit_constraints,  # Use filtered constraints
        "missing_information": parsed.get("missing_information", []),
        "confidence_score": adjusted_confidence,  # Use adjusted confidence
        "decision_type": decision_type,
    })
