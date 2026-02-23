## CareerCompass — A Multi-Agent Career Planning Assistant
Student Planning Assistant — Multi-Agent LangGraph System

A multi-agent system built with **LangGraph** that handles student career planning queries through a linear pipeline of three specialized agents communicating exclusively via shared state.

---

## Overview

A student submits a career planning question (e.g., *"Should I focus on ML or backend development?"*). The system processes the query through three sequential agents:

1. **Domain Reasoning Agent** — interprets the query, extracts goals and constraints, detects decision type (comparison vs. direct path), and assigns a confidence score.
2. **Recommendation Agent** — generates structured recommendations based on decision type: either a 2-week exploration experiment or a direct career path recommendation.
3. **Coordinator Agent** — compiles the final response, adapting output format based on decision type.

---

## Architecture

```
┌──────────────┐
│ Student Query│
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────┐
│ Domain Reasoning Agent              │
│ • Parse and interpret goal          │
│ • Extract constraints               │
│ • Classify decision type            │
│ • Score confidence                  │
└──────────┬────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│ Recommendation Agent                 │
│ • Comparison path (exploration plan) │
│ • OR Direct path (career guidance)   │
└──────────┬────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│ Coordinator Agent                    │
│ • Format final response              │
│ • Adapt based on decision type       │
└──────────┬────────────────────────────┘
           │
           ▼
        Output
```

---

## Agent Responsibilities

| Agent | Reads | Writes | Logic |
|---|---|---|---|
| **Domain Reasoning** | `user_input` | `interpreted_goal`, `extracted_constraints`, `missing_information`, `confidence_score`, `decision_type` | Uses LLM to parse query; detects "comparison" or "direct_path" keywords; adjusts confidence based on context availability |
| **Recommendation** | `interpreted_goal`, `extracted_constraints`, `missing_information`, `decision_type` | `recommendations` | For comparison: generates 2-week exploration plan with two parallel focuses; for direct_path: generates primary/secondary career paths |
| **Coordinator** | `recommendations`, `decision_type`, `interpreted_goal`, `confidence_score` | `final_response` | For comparison: formats exploration plan deterministically; for direct_path: uses LLM to synthesize friendly response |

---

## Decision Type Classification

The Domain Reasoning Agent automatically classifies queries as:

- **Comparison**: Query contains keywords like "or", "vs", "which", "better", "compare"
  - Generates a 2-week structured exploration experiment
  - Allows student to test both paths before committing

- **Direct Path**: All other queries
  - Generates a focused career recommendation
  - Provides immediate next steps

---

## How Recommendations Differ by Decision Type

### Comparison Decisions

The Recommendation Agent generates:
- `exploration_plan`: Two items (Week 1 and Week 2) with specific focus and task
- `decision_criteria`: Three reflective questions to evaluate after exploration
- `final_suggestion`: Guidance on how to choose after testing
- `resources`: Learning materials for both paths

The Coordinator formats this deterministically (no LLM) to ensure consistent structure.

### Direct Path Decisions

The Recommendation Agent generates:
- `primary_path`: Most suitable career direction
- `secondary_path`: Viable alternative
- `reasoning`: Why primary_path fits the student's context
- `immediate_next_steps`: Three concrete actions for this week
- `resources`: Learning materials and links

The Coordinator uses the LLM to synthesize these into an encouraging, personalized response.

---

## Workflow Execution

1. **Initialize**: `AssistantState(user_input=query)`
2. **Domain Reasoning runs**: Parses query, sets `interpreted_goal`, `extracted_constraints`, `confidence_score`, `decision_type`
3. **Recommendation runs**: Reads decision_type and generates appropriate `recommendations` structure
4. **Coordinator runs**: Formats final response based on decision_type
5. **Output**: `final_response` is printed to user

The workflow is intentionally linear. All three agents execute sequentially, with decision-specific branching handled inside the Recommendation Agent rather than via conditional graph edges.
---

## Running the Project

### Prerequisites

- Python 3.8+
- Ollama running locally with `llama2` model available

### Installation

```bash
pip install -r requirements.txt
```

### Running

Interactive mode:
```bash
python main.py
```

With a query argument:
```bash
python main.py --query "Should I focus on ML or backend development?"
```

### Output

The system prints:
1. The parsed goal and extracted constraints
2. The confidence score and decision type
3. Structured recommendations or next steps
4. Final response tailored to the query

---
## Example
<img width="1099" height="809" alt="image" src="https://github.com/user-attachments/assets/fb181961-9b40-4694-982e-5f2675fc44c6" />


## Project Structure

```
assignment/
│
├── agents/
│   ├── __init__.py
│   ├── coordinator.py        # Compiles final response based on decision type
│   ├── domain_reasoning.py   # Parses intent, extracts constraints, scores confidence
│   └── recommendation.py     # Generates structure recommendations for each decision type
│
├── graph/
│   ├── __init__.py
│   └── workflow.py           # LangGraph StateGraph definition
│
├── state/
│   ├── __init__.py
│   └── schema.py             # AssistantState Pydantic model (shared state)
│
├── main.py                   # CLI entry point
├── requirements.txt
└── README.md
```

---

## Design Details

**State-driven architecture**: Agents are orchestrated by state transitions, not by direct function calls. This decouples implementation and allows agents to be developed independently.

**Pydantic state schema**: `AssistantState` uses Pydantic for field validation and documentation. Node functions convert between dict (LangGraph) and Pydantic models via `model_dump()` and `model_validate()`.

**Decision-type branching in recommendation logic**: Rather than conditional graph edges, branching occurs inside the Recommendation Agent—it reads `decision_type` and selects the appropriate generation strategy. This keeps the graph linear and the branching logic explicit.

**Deterministic comparison formatting**: When handling comparison decisions, the Coordinator formats output directly without LLM calls, ensuring reproducible structure for exploration plans.

**LLM with temperature=0**: The system uses Ollama's llama2 model with deterministic temperature for consistent JSON parsing in structured outputs.
