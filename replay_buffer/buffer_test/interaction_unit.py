from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any


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
class UserFeedback:
    """
    User judgement signal.
    """
    type: UserFeedbackType


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

    # Full interaction trajectory (raw, ordered)
    trajectory: List[Any]

    # Single unified user signal
    user_feedback: UserFeedback

    # Learning signal (ONE sentence or None)
    insight: Optional[str]

    timestamp: str
