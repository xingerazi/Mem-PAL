"""
interaction_unit.py

Final Interaction Unit (IU) schema (scene removed).

Principles:
- Minimal but sufficient fields
- topic + user_query fully define context
- Single unified user_feedback signal
- agent_actions is a sequence
- insight diagnoses personalization / memory / reasoning behavior
"""

from enum import Enum
from typing import List, Optional, Dict
from dataclasses import dataclass
from datetime import datetime


# =========================
# Enums
# =========================

class AgentActionType(str, Enum):
    TOOL_CALL = "tool_call"
    RESPOND = "respond"


class UserFeedbackType(str, Enum):
    CONFIRM = "confirm"
    REJECT = "reject"
    REVISE = "revise"


class InsightType(str, Enum):
    MEMORY_MISS = "memory_miss"
    MEMORY_OVERUSE = "memory_overuse"
    TEMPORAL_MISWEIGHT = "temporal_misweight"
    PROFILE_CONFLICT = "profile_conflict"
    PROFILE_STALE = "profile_stale"
    CORRECT_MEMORY_USAGE = "correct_memory_usage"
    CORRECT_REASONING_PATH = "correct_reasoning_path"


# =========================
# Core Data Structures
# =========================

@dataclass
class AgentAction:
    type: AgentActionType
    tool: Optional[str] = None
    tool_args: Optional[Dict] = None
    observation: Optional[str] = None


@dataclass
class UserFeedback:
    """
    Unified user signal:
    - judgement (confirm / reject / revise)
    - may implicitly contain new user information
    """
    type: UserFeedbackType
    content: str


@dataclass
class Insight:
    """
    Learning signal for personalization.
    """
    type: InsightType
    description: str
    affected_memory: Optional[List[str]] = None
    weight_update: Optional[Dict[str, List[str]]] = None


@dataclass
class InteractionUnit:
    """
    Minimal replayable unit for personalization learning.
    """
    iu_id: str

    # coarse semantic scope
    topic: str

    # fine-grained, natural language context
    user_query: str

    agent_actions: List[AgentAction]
    agent_reply: str

    # single user judgement + information carrier
    user_feedback: UserFeedback

    success: bool
    insight: Optional[Insight]

    timestamp: str


# =========================
# Example 1 — REJECT
# =========================

def example_reject_case() -> InteractionUnit:
    return InteractionUnit(
        iu_id="iu_20260103_901",

        topic="diet_recommendation",
        user_query="我今天晚饭吃什么比较好？",

        agent_actions=[
            AgentAction(
                type=AgentActionType.TOOL_CALL,
                tool="retrieve_user_memory",
                tool_args={"query": "food_preference"},
                observation="User frequently ate ramen in the past"
            ),
            AgentAction(type=AgentActionType.RESPOND)
        ],

        agent_reply="你以前很常吃拉面，可以继续吃拉面。",

        user_feedback=UserFeedback(
            type=UserFeedbackType.REJECT,
            content="我最近在减肥，我要吃鸡胸肉。"
        ),

        success=False,

        insight=Insight(
            type=InsightType.TEMPORAL_MISWEIGHT,
            description="近期健康目标未被纳入决策，历史饮食偏好被过度使用。",
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
    return InteractionUnit(
        iu_id="iu_20260103_902",

        topic="travel_planning",
        user_query="我这个周末能不能去英国玩？",

        agent_actions=[
            AgentAction(
                type=AgentActionType.TOOL_CALL,
                tool="retrieve_user_profile",
                tool_args={"query": "visa_status"},
                observation="User holds a Schengen short-term visa"
            ),
            AgentAction(
                type=AgentActionType.TOOL_CALL,
                tool="check_destination_requirement",
                tool_args={"destination": "UK"},
                observation="UK requires separate visa"
            ),
            AgentAction(type=AgentActionType.RESPOND)
        ],

        agent_reply="你目前只有申根签证，不适合去英国，建议在申根区内旅行。",

        user_feedback=UserFeedback(
            type=UserFeedbackType.CONFIRM,
            content="对，我确实只有申根签证。"
        ),

        success=True,

        insight=Insight(
            type=InsightType.CORRECT_REASONING_PATH,
            description="正确检索用户签证状态并结合目的地规则完成推理。",
            affected_memory=[
                "user_visa_status: schengen_short_term"
            ]
        ),

        timestamp=datetime.utcnow().isoformat()
    )


# =========================
# Example 3 — REVISE
# =========================

def example_revise_case() -> InteractionUnit:
    return InteractionUnit(
        iu_id="iu_20260103_903",

        topic="course_planning",
        user_query="我下学期该选什么课？",

        agent_actions=[
            AgentAction(
                type=AgentActionType.TOOL_CALL,
                tool="retrieve_user_profile",
                tool_args={"query": "academic_background"},
                observation="User studied linear algebra before"
            ),
            AgentAction(type=AgentActionType.RESPOND)
        ],

        agent_reply="你之前学过线性代数，可以选《数值分析》。",

        user_feedback=UserFeedback(
            type=UserFeedbackType.REVISE,
            content="数值分析不太对，我是偏机器学习方向，应该选《概率论》。"
        ),

        success=True,

        insight=Insight(
            type=InsightType.MEMORY_OVERUSE,
            description="过度依赖基础课程经历，忽略了用户当前的专业方向目标。",
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
