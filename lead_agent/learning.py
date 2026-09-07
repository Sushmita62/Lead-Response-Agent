"""
Explainable learning mechanism for opener styles.

The agent experiments with three opener styles:

- direct_confirm
- warm_context
- question_first

An epsilon-greedy bandit balances exploration and exploitation.

Historical conversations seed the bandit.
Live conversations update it when a meaningful outcome is observed.

Important:
A lead simply replying is NOT automatically a success.
STOP, rejection, or disengagement are negative outcomes.
Handoff and meaningful qualification progress are positive outcomes.
"""

import random


EPSILON = 0.2

ARMS = [
    "direct_confirm",
    "warm_context",
    "question_first",
]


# ---------------------------------------------------------------------
# Bandit
# ---------------------------------------------------------------------

class Bandit:

    def __init__(self):
        self.stats = {
            arm: {
                "pulls": 0,
                "successes": 0,
            }
            for arm in ARMS
        }

    def choose(self):
        """
        Epsilon-greedy selection.

        20% of the time:
            explore a random opener.

        Otherwise:
            use the best observed opener.
        """

        if (
            random.random() < EPSILON
            or all(
                value["pulls"] == 0
                for value in self.stats.values()
            )
        ):
            return random.choice(ARMS)

        def rate(arm):
            value = self.stats[arm]

            if value["pulls"] == 0:
                return -1

            return (
                value["successes"]
                / value["pulls"]
            )

        return max(
            ARMS,
            key=lambda arm: (
                rate(arm),
                -self.stats[arm]["pulls"],
            ),
        )

    def record(self, arm, success: bool):

        if arm not in self.stats:
            return

        self.stats[arm]["pulls"] += 1

        if success:
            self.stats[arm]["successes"] += 1

    def snapshot(self):

        output = {}

        for arm, value in self.stats.items():

            if value["pulls"]:
                win_rate = round(
                    value["successes"]
                    / value["pulls"],
                    2,
                )
            else:
                win_rate = None

            output[arm] = {
                **value,
                "win_rate": win_rate,
            }

        return output


# ---------------------------------------------------------------------
# Historical opener classification
# ---------------------------------------------------------------------

def _classify_opener(text: str) -> str:

    text = (text or "").strip().lower()

    if any(
        phrase in text
        for phrase in [
            "still on your list",
            "still interested",
            "still available",
        ]
    ):
        return "direct_confirm"

    if (
        text.endswith("?")
        and len(text) < 80
    ):
        return "question_first"

    return "warm_context"


# ---------------------------------------------------------------------
# Historical outcome classification
# ---------------------------------------------------------------------

def _historical_success(conversation: dict) -> bool:
    """
    Determine whether a historical conversation represents a positive
    learning signal.

    Explicit negative signals take priority over generic replies.
    """

    outcome = str(
        conversation.get(
            "outcome",
            "",
        )
    ).lower().strip()

    negative_outcomes = {
        "stop",
        "unsubscribe",
        "negative",
        "disengaged",
        "abusive",
        "bad",
    }

    if outcome in negative_outcomes:
        return False

    positive_outcomes = {
        "handoff_showing",
        "handoff_call",
        "qualified",
        "appointment",
        "showing",
    }

    if outcome in positive_outcomes:
        return True

    turns = conversation.get(
        "turns",
        [],
    )

    lead_messages = [
        turn
        for turn in turns
        if turn.get("from") == "lead"
    ]

    if not lead_messages:
        return False

    combined = " ".join(
        str(turn.get("text", ""))
        for turn in lead_messages
    ).lower()

    negative_phrases = [
        "stop",
        "unsubscribe",
        "don't contact",
        "do not contact",
        "leave me alone",
        "not interested",
        "remove me",
    ]

    if any(
        phrase in combined
        for phrase in negative_phrases
    ):
        return False

    # A continuing conversation is a positive signal when no
    # explicit negative outcome is present.
    return True


# ---------------------------------------------------------------------
# Seed from historical conversations
# ---------------------------------------------------------------------

def seed_from_past_conversations(
    bandit: Bandit,
    past_conversations,
):
    """
    Seed the bandit once from the supplied historical conversations.
    """

    # Prevent duplicate seeding across Streamlit reruns.
    if any(
        value["pulls"] > 0
        for value in bandit.stats.values()
    ):
        return

    for conversation in past_conversations:

        turns = conversation.get(
            "turns",
            [],
        )

        if not turns:
            continue

        first_agent_turn = next(
            (
                turn
                for turn in turns
                if turn.get("from") == "agent"
            ),
            None,
        )

        if not first_agent_turn:
            continue

        style = _classify_opener(
            first_agent_turn.get(
                "text",
                "",
            )
        )

        success = _historical_success(
            conversation
        )

        bandit.record(
            style,
            success,
        )


# ---------------------------------------------------------------------
# Live outcome classification
# ---------------------------------------------------------------------

def classify_live_outcome(
    prior_state: dict,
    new_state: dict,
):
    """
    Return:

        True
            Positive learning signal.

        False
            Negative learning signal.

        None
            No sufficiently clear outcome yet.
    """

    decision = new_state.get(
        "decision"
    )

    intent = new_state.get(
        "intent",
        {},
    )

    current_message = str(
        new_state.get(
            "message",
            "",
        )
    ).lower()

    # -------------------------------------------------------------
    # Strong negative signals
    # -------------------------------------------------------------

    if intent.get("stop_request"):
        return False

    negative_phrases = [
        "stop",
        "unsubscribe",
        "don't contact",
        "do not contact",
        "leave me alone",
        "not interested",
        "remove me",
    ]

    if any(
        phrase in current_message
        for phrase in negative_phrases
    ):
        return False

    if decision == "stop":
        return False

    # -------------------------------------------------------------
    # Strong positive signal
    # -------------------------------------------------------------

    if decision == "handoff":
        return True

    # -------------------------------------------------------------
    # Qualification progress
    # -------------------------------------------------------------

    before = prior_state.get(
        "qualification",
        {},
    )

    after = new_state.get(
        "qualification",
        {},
    )

    qualification_fields = [
        "timeline",
        "budget",
        "financing",
        "area",
        "motivation",
    ]

    newly_known = 0

    for field in qualification_fields:

        old_value = before.get(field)
        new_value = after.get(field)

        if (
            not old_value
            and new_value
        ):
            newly_known += 1

    if newly_known > 0:
        return True

    # -------------------------------------------------------------
    # Concrete conversational progress
    # -------------------------------------------------------------

    if intent.get("concrete_signals"):
        return True

    if intent.get("questions"):
        return True

    # No reliable signal yet.
    return None


# ---------------------------------------------------------------------
# Live bandit update
# ---------------------------------------------------------------------

def record_outcome_from_turn(
    prior_state: dict,
    new_state: dict,
):
    """
    Record the opener's outcome once a meaningful outcome is observed.

    We deliberately do NOT store a generic permanent
    `learning_recorded=True` flag before an outcome exists.

    `opener_outcome_recorded` is set only after the opener has actually
    received a positive or negative learning signal. This allows neutral
    turns to remain unscored and prevents the same opener from being
    counted repeatedly.
    """

    opener_style = (
        prior_state.get("opener_style")
        or new_state.get("opener_style")
    )

    if not opener_style:
        return

    # Already scored for this lead.
    if new_state.get(
        "opener_outcome_recorded",
        False,
    ):
        return

    outcome = classify_live_outcome(
        prior_state,
        new_state,
    )

    # No meaningful signal yet.
    if outcome is None:
        return

    BANDIT.record(
        opener_style,
        success=outcome,
    )

    # This flag is intentionally persistent:
    # one opener gets one outcome score.
    new_state["opener_outcome_recorded"] = True


# ---------------------------------------------------------------------
# Global bandit
# ---------------------------------------------------------------------

BANDIT = Bandit()