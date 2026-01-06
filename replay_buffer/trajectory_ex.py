"""
interaction_unit.py

Final Interaction Unit (IU) schema.

Design principles:
- NO InsightType / NO enum-based insight classification
- Insight is free-form diagnosis for memory / profile / reasoning errors
- AgentAction ONLY represents tool calls
- agent_reply is the ONLY user-facing output
- Designed for personalization + retrieval + RL systems
"""

from enum import Enum
from typing import List, Optional, Dict
from dataclasses import dataclass
from datetime import datetime


# =========================
# Enums (ONLY essential ones)
# =========================

class UserFeedbackType(str, Enum):
    """
    Unified user feedback signal.
    """
    CONFIRM = "confirm"
    REJECT = "reject"
    REVISE = "revise"


# =========================
# Core Data Structures
# =========================

@dataclass
class AgentAction:
    """
    One atomic tool call made by the agent.

    NOTE:
    - This class ONLY represents tool usage
    - Final natural language response is stored in `agent_reply`
    """
    tool: str
    tool_args: Optional[Dict] = None
    observation: Optional[str] = None


@dataclass
class UserFeedback:
    """
    User judgement + implicit new information.
    """
    type: UserFeedbackType
    content: str


@dataclass
class Insight:
    """
    Free-form diagnosis derived from user feedback.

    Rules:
    - NO predefined categories
    - NO dialogue strategy advice
    - Only diagnose memory / profile / weighting / reasoning issues
    """
    description: str

    affected_memory: Optional[List[str]] = None
    weight_update: Optional[Dict[str, List[str]]] = None


@dataclass
class InteractionUnit:
    """
    Minimal replayable unit for personalization learning.
    """
    iu_id: str

    # Coarse semantic scope
    topic: str

    # Fine-grained natural language context
    user_query: str

    # Tool usage trace (can be empty list)
    agent_actions: List[AgentAction]

    # Final user-facing response
    agent_reply: str

    # Single unified user signal
    user_feedback: UserFeedback

    # Whether the final decision is accepted
    success: bool

    # Optional learning signal
    insight: Optional[Insight]

    timestamp: str


# =========================
# Example 1 — REJECT
# =========================

def example_reject_case() -> InteractionUnit:
    """
    Agent overuses historical food preference
    and ignores recent health constraint.
    """

    return InteractionUnit(
        iu_id="iu_20260103_1001",

        topic="diet_recommendation",
        user_query="我今天晚饭吃什么比较好？",

        agent_actions=[
            AgentAction(
                tool="retrieve_user_memory",
                tool_args={"query": "food_preference"},
                observation="User frequently ate ramen in the past"
            )
        ],

        agent_reply="你以前很常吃拉面，可以继续吃拉面。",

        user_feedback=UserFeedback(
            type=UserFeedbackType.REJECT,
            content="我最近在减肥，我要吃鸡胸肉。"
        ),

        success=False,

        insight=Insight(
            description=(
                "近期健康目标未被纳入决策，"
                "历史饮食偏好被过度依赖。"
            ),
            affected_memory=[
                "recent_health_goal: weight_loss",
                "historical_food_preference: ramen"
            ],
            weight_update={
                "increase": ["recent_health_goal"],
                "decrease": ["historical_food_preference"]
            }
        ),

        timestamp=datetime.utcnow().isoformat()
    )


# =========================
# Example 2 — CONFIRM
# =========================

def example_confirm_case() -> InteractionUnit:
    """
    Agent correctly reasons over visa constraints.
    No insight is strictly required.
    """

    return InteractionUnit(
        iu_id="iu_20260103_1002",

        topic="travel_planning",
        user_query="我这个周末能不能去英国玩？",

        agent_actions=[
            AgentAction(
                tool="retrieve_user_profile",
                tool_args={"query": "visa_status"},
                observation="User holds a Schengen short-term visa"
            ),
            AgentAction(
                tool="check_destination_requirement",
                tool_args={"destination": "UK"},
                observation="UK requires separate visa"
            )
        ],

        agent_reply="你目前只有申根签证，不适合去英国，建议在申根区内旅行。",

        user_feedback=UserFeedback(
            type=UserFeedbackType.CONFIRM,
            content="对，我确实只有申根签证。"
        ),

        success=True,

        insight=None,

        timestamp=datetime.utcnow().isoformat()
    )


# =========================
# Example 3 — REVISE
# =========================

def example_revise_case() -> InteractionUnit:
    """
    Decision framework is correct,
    but personalization evidence is corrected by user.
    """

    return InteractionUnit(
        iu_id="iu_20260103_1003",

        topic="course_planning",
        user_query="我下学期该选什么课？",

        agent_actions=[
            AgentAction(
                tool="retrieve_user_profile",
                tool_args={"query": "academic_background"},
                observation="User studied linear algebra before"
            )
        ],

        agent_reply="你之前学过线性代数，可以选《数值分析》。",

        user_feedback=UserFeedback(
            type=UserFeedbackType.REVISE,
            content="数值分析不太对，我是偏机器学习方向，应该选《概率论》。"
        ),

        success=True,

        insight=Insight(
            description=(
                "过度依赖已修课程经历，"
                "忽略了用户当前的专业方向目标。"
            ),
            affected_memory=[
                "completed_course: linear_algebra",
                "intended_specialization: machine_learning"
            ],
            weight_update={
                "increase": ["intended_specialization"],
                "decrease": ["completed_course"]
            }
        ),

        timestamp=datetime.utcnow().isoformat()
    )


if __name__ == "__main__":
    print(example_reject_case())
    print(example_confirm_case())
    print(example_revise_case())
