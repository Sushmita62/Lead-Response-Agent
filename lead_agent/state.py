"""
The state object that flows through every LangGraph node.

It contains the information produced or needed during the current graph run.
Persistent conversation and lead data are stored separately in SQLite.
Static case-study data is loaded by data_loader.py and accessed by tools.

Keeping the state explicit makes the graph inspectable: at any point,
the evaluator can see what the previous nodes understood, decided, and produced.
"""

from typing import TypedDict, List, Dict, Optional, Literal


class LeadState(TypedDict, total=False):
    # Lead and current conversation
    lead_id: str
    lead: dict                      # raw lead record from sample_leads.json
    message: str                    # incoming lead message this turn
    history: List[Dict]             # [{"role": "lead"/"agent", "text": str, "day": int}]
    day: int                        # simulated day this turn happens on

    # Lead qualification
    qualification: Dict[str, Optional[str]]
    # timeline / budget / financing / area / motivation

    # Nurture state
    nurture: Dict
    # {attempt_count, next_contact_at, stopped}

    # Safety check node output
    safety_flags: List[str]
    # e.g. ["injection_attempt", "fair_housing_steering"]

    safety_block: Optional[str]
    # set if the turn must be hard-stopped before an LLM reply

    # Understand-intent node output
    intent: Dict
    # {"category": str,
    #  "needs_property_data": bool,
    #  "qualification_updates": dict}

    # Property/market tool output
    tool_results: Dict
    # {"listing": dict|None, "data_flags": [str]}

    # Decision-agent node output
    decision: Literal["reply", "handoff", "nurture", "stop"]
    decision_reason: str

    # Action node output
    draft_reply: str
    opener_style: Optional[str]
    # chosen once, on the first turn

    # Response-guardrail node output
    final_reply: str
    guardrail_flags: List[str]
    guardrail_blocked: bool

    # Bookkeeping / evaluator visibility
    handoff_ready: bool
    stop_outreach: bool
    reasoning_log: List[str]
    # one line per node, for the evaluator UI

    abuse_count: int
    # cross-turn counter so a second abusive message actually triggers
    # the deterministic disengage-on-repeat policy in guardrails.py

    opener_outcome_recorded: bool

    fair_housing_redirect: bool
    # set deterministically by decision_node when input_safety_check flags
    # fair_housing_steering; must be declared here or LangGraph silently
    # drops it before reply_node ever sees it (same failure mode as
    # abuse_count above)

    active_listing_id: Optional[str]
    # the listing currently "in focus" for this conversation -- updated
    # whenever a property_id/address_hint resolves to exactly one real
    # listing, and used as the fallback when the lead refers to "this
    # property"/"it" with no explicit address on a later turn. Without
    # this, only the very first turn could resolve pronoun references;
    # turn 2+ would incorrectly come back empty.
    # set once the opener-style bandit has been scored for this lead,
    # so the same opener isn't double-counted across turns