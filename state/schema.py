from typing import Optional
from pydantic import BaseModel, Field


class AssistantState(BaseModel):
    """
    This is the single source of truth passed through every node in the graph.
    Agents read from and write to this state; they never call each other directly.
    """

    user_input: str = Field(
        description="The original query from the student."
    )

    # --- Domain Reasoning Agent outputs ---
    interpreted_goal: Optional[str] = Field(
        default=None,
        description="The agent's interpretation of what the student is trying to achieve."
    )
    extracted_constraints: Optional[list[str]] = Field(
        default=None,
        description="Constraints or preferences extracted from the student's input."
    )
    missing_information: Optional[list[str]] = Field(
        default=None,
        description="Information that would improve the quality of recommendations."
    )
    confidence_score: Optional[float] = Field(
        default=None,
        description="How confident the Domain Reasoning Agent is in its interpretation (0.0–1.0)."
    )

    # --- Decision type classification ---
    decision_type: Optional[str] = Field(
        default=None,
        description="Classification of the decision type: 'comparison' or 'direct_path'."
    )

    # --- Recommendation Agent outputs ---
    recommendations: Optional[dict] = Field(
        default=None,
        description="Structured recommendations produced for the student."
    )

    # --- Final output ---
    final_response: Optional[str] = Field(
        default=None,
        description="The consolidated response returned to the student."
    )
