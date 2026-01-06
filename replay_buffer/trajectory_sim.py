"""
interaction_unit.py

Final Interaction Unit (IU) schema.

Design principles:
- Insight is a SINGLE free-form string (or None)
- No structured insight fields
- No enum-based insight classification
- AgentAction ONLY represents tool calls
- agent_reply is the ONLY user-facing output
- Designed for personalization + retrieval + RL systems
"""

from enum import Enum
from typing import List, Optional, Dict
from dataclasses import dataclass
from datetime import datetime
from datetime import datetime
from typing import List


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
    """
    tool: str
    tool_args: Optional[Dict] = None
    observation: Optional[str] = None

@dataclass
class IUCausalLink:
    """
    Directed causal relation between InteractionUnits.
    """
    from_iu_id: str
    to_iu_id: str
    relation: Optional[str] = None
    timestamp: Optional[str] = None

@dataclass
class UserFeedback:
    """
    User judgement + implicit new information.
    """
    type: UserFeedbackType
    content: str


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

    # Tool usage trace (can be empty)
    agent_actions: List[AgentAction]

    # Final user-facing response
    agent_reply: str

    # Single unified user signal
    user_feedback: UserFeedback

    # Learning signal (ONE sentence or None)
    insight: Optional[str]

    timestamp: str



# =========================
# Pretty Renderer
# =========================

def render_interaction_unit(iu: InteractionUnit) -> str:
    lines = []

    status_icon = "✅ SUCCESS" if iu.success else "❌ FAILURE"

    lines.append("─" * 48)
    lines.append(f"🧩 InteractionUnit: {iu.iu_id}")
    lines.append(f"📌 Topic: {iu.topic}")
    lines.append(f"📊 Status: {status_icon}")
    lines.append("─" * 48)

    lines.append("🙋 User Query")
    lines.append(f"  {iu.user_query}")

    lines.append("\n🤖 Agent Actions")
    if iu.agent_actions:
        for idx, act in enumerate(iu.agent_actions, 1):
            lines.append(f"  {idx}. Tool: {act.tool}")
            lines.append(f"     Args: {act.tool_args}")
            lines.append(f"     Observation: {act.observation}")
    else:
        lines.append("  (none)")

    lines.append("\n💬 Agent Reply")
    lines.append(f"  {iu.agent_reply}")

    lines.append("\n🧠 User Feedback")
    lines.append(f"  Type: {iu.user_feedback.type.value.upper()}")
    lines.append(f"  Content: {iu.user_feedback.content}")

    lines.append("\n🔍 Insight")
    lines.append(f"  {iu.insight if iu.insight else 'None'}")

    lines.append("\n⏱ Timestamp")
    lines.append(f"  {iu.timestamp}")

    return "\n".join(lines)


def render_trajectory(units: List[InteractionUnit]) -> None:
    for iu in units:
        print(render_interaction_unit(iu))
        print()  # spacing


# =========================
# Example 1 — REJECT
# =========================

def example_reject_case() -> InteractionUnit:
    """
    Agent ignores recent health constraint.
    """

    return InteractionUnit(
        iu_id="iu_20260103_2001",

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

        insight="使用了历史饮食偏好，但忽略了用户近期的减脂目标。",

        timestamp=datetime.utcnow().isoformat()
    )


# =========================
# Example 2 — CONFIRM
# =========================

def example_confirm_case() -> InteractionUnit:
    """
    Agent correctly reasons over visa constraints.
    """

    return InteractionUnit(
        iu_id="iu_20260103_2002",

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
    Decision direction is correct, but evidence is corrected by user.
    """

    return InteractionUnit(
        iu_id="iu_20260103_2003",

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

        insight="过度依据已修课程背景，未优先考虑用户当前的专业方向目标。",

        timestamp=datetime.utcnow().isoformat()
    )


# =========================
# Main
# =========================

if __name__ == "__main__":
    trajectory = [
        example_reject_case(),
        example_confirm_case(),
        example_revise_case()
    ]

    render_trajectory(trajectory)