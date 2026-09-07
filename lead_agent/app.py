"""
The front door. Run with: streamlit run app.py
Every lead message goes in through here, runs the graph, and the sidebar
shows everything the graph decided along the way.
"""

import streamlit as st
from graph import (
    build_graph,
    load_state_node,
    run_nurture_followup,
    response_guardrail_node,
)
from data_loader import DB
import storage
import nurture
from learning import (
    BANDIT,
    seed_from_past_conversations,
    record_outcome_from_turn,
)
import llm

st.set_page_config(page_title="Lead Response Agent", layout="wide")


# ---------------------------------------------------------
# Initialize
# ---------------------------------------------------------

if "seeded" not in st.session_state:
    seed_from_past_conversations(
        BANDIT,
        DB["past_conversations"]
    )
    st.session_state["seeded"] = True

if "graph" not in st.session_state:
    st.session_state["graph"] = build_graph()

GRAPH = st.session_state["graph"]


def lead_by_id(lead_id):
    return next(
        (l for l in DB["leads"] if l["lead_id"] == lead_id),
        None
    )


def get_or_init_state(lead_id):
    saved = storage.load_state(lead_id)

    if saved:
        return saved

    lead = lead_by_id(lead_id)

    return {
        "lead_id": lead_id,
        "lead": lead,
        "history": [],
        "qualification": {
            f: None
            for f in [
                "timeline",
                "budget",
                "financing",
                "area",
                "motivation"
            ]
        },
        "nurture": {
            "attempt_count": 0,
            "next_contact_at": None,
            "stopped": False
        },
        "reasoning_log": [],
        "safety_flags": [],
        "guardrail_flags": [],
        "handoff_ready": False,
        "stop_outreach": False,
    }


def run_nurture_tick(lead_id, day):
    """Runs a scheduled, automated nurture check-in -- NOT a full graph
    turn, since there's no real lead message to interpret on a tick."""
    prior = get_or_init_state(lead_id)
    lead = lead_by_id(lead_id)
    state = {**prior, "lead_id": lead_id, "lead": lead, "day": day}
    state = load_state_node(state)
    state = run_nurture_followup(state)

    if state.get("draft_reply"):
        state = response_guardrail_node(state)
        state.setdefault("history", []).append({
            "role": "agent", "text": state["final_reply"], "day": day,
        })
    else:
        state["final_reply"] = None

    storage.save_state(lead_id, state, day)
    return state


def run_turn(lead_id, message, day, is_nurture=False):
    prior = get_or_init_state(lead_id)
    lead = lead_by_id(lead_id)

    turn_state = {
        **prior,
        "lead_id": lead_id,
        "lead": lead,
        "message": message,
        "day": day,
        "is_nurture": is_nurture,
    }

    out = GRAPH.invoke(turn_state)

    # Do not learn from automated nurture messages.
    if not is_nurture:
        record_outcome_from_turn(prior, out)
        # graph.py's save_node already persisted `out` once, before this
        # flag existed -- re-save so opener_outcome_recorded actually
        # sticks and the same opener isn't double-scored next turn.
        storage.save_state(lead_id, out, day)

    return out


# ----------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------

st.sidebar.title("Lead Response Agent")
st.sidebar.caption(f"LLM: {llm.current_mode_label()}")

lead_options = {
    f"{l['lead_id']} — {l['name']}": l["lead_id"]
    for l in DB["leads"]
}

choice = st.sidebar.selectbox(
    "Select lead",
    list(lead_options.keys())
)

lead_id = lead_options[choice]

sim_day = storage.get_sim_day()
st.sidebar.markdown(f"**Simulated day:** {sim_day}")


# ---------------------------------------------------------
# Simulated time / nurture
# ---------------------------------------------------------

c1, c2, c3 = st.sidebar.columns(3)

for col, days in zip((c1, c2, c3), (1, 7, 30)):

    if col.button(f"+{days}d"):

        new_day = sim_day + days
        storage.set_sim_day(new_day)

        fired = []

        for lid in storage.all_lead_ids_with_state():

            s = storage.load_state(lid)

            if s and nurture.is_due(
                s.get("nurture", {}),
                new_day
            ):
                run_nurture_tick(lid, new_day)
                fired.append(lid)

        st.session_state["last_nurture_sweep"] = fired
        st.rerun()


if st.session_state.get("last_nurture_sweep") is not None:
    fired = st.session_state["last_nurture_sweep"]

    st.sidebar.info(
        f"Nurture fired for: "
        f"{', '.join(fired) if fired else 'no one due'}"
    )


# ---------------------------------------------------------
# Reset
# ---------------------------------------------------------

if st.sidebar.button("Reset all data"):
    storage.reset_all()
    st.session_state.pop("last_nurture_sweep", None)
    st.rerun()


# ---------------------------------------------------------
# Learning
# ---------------------------------------------------------

st.sidebar.divider()
st.sidebar.subheader("Learning — opener style")

snap = BANDIT.snapshot()

for arm, v in snap.items():

    wr = (
        "n/a"
        if v["win_rate"] is None
        else f"{int(v['win_rate'] * 100)}%"
    )

    st.sidebar.progress(
        v["win_rate"] or 0,
        text=f"{arm}: {v['pulls']} pulls, {wr}"
    )


# ---------------------------------------------------------
# Data quality
# ---------------------------------------------------------

st.sidebar.divider()
st.sidebar.subheader("Global data-quality flags")

for f in DB["realtor_flags"]:
    st.sidebar.markdown(f"⚠ realtor: {f}")

for lst in DB["listings"]:
    for f in lst["flags"]:
        st.sidebar.markdown(
            f"⚠ {lst['listing_id']}: {f}"
        )


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

state = get_or_init_state(lead_id)
lead = lead_by_id(lead_id)

col_chat, col_eval = st.columns([2, 1])


# ================================================================
# Chat
# ================================================================

with col_chat:

    st.subheader(f"{lead['name']} — {lead_id}")

    st.caption(
        f"Source: {lead['source']} · "
        f"Inquiry: {lead['initial_inquiry']['text']}"
    )

    if not state["history"]:

        if st.button(
            "Start conversation (send opening message)"
        ):
            run_turn(
                lead_id,
                lead["initial_inquiry"]["text"],
                sim_day
            )
            st.rerun()

    else:

        for t in state["history"]:

            with st.chat_message(
                "assistant"
                if t["role"] == "agent"
                else "user"
            ):
                st.write(t["text"])

        if "handoff_banner_seen" not in st.session_state:
            st.session_state["handoff_banner_seen"] = set()

        if state.get("handoff_ready") and lead_id not in st.session_state["handoff_banner_seen"]:
            st.info(
                "This lead has been passed to the human realtor for "
                "follow-up. You can keep chatting here in the meantime."
            )
            st.session_state["handoff_banner_seen"].add(lead_id)

        if (
            not state["nurture"].get("stopped")
            and not state.get("stop_outreach")
        ):

            msg = st.chat_input("Type as the lead…")

            if msg:
                with st.chat_message("user"):
                    st.write(msg)
                with st.chat_message("assistant"):
                    with st.spinner("Mira is typing…"):
                        run_turn(
                            lead_id,
                            msg,
                            sim_day
                        )
                st.rerun()

        elif (
            state["nurture"].get("stopped")
            or state.get("stop_outreach")
        ):

            st.warning(
                "Outreach stopped for this lead."
            )


# ================================================================
# Evaluator panel
# ================================================================

with col_eval:

    st.subheader("Evaluator panel")

    st.markdown("**Decision**")
    st.write(state.get("decision", "—"))

    st.markdown("**Reason**")
    st.write(state.get("decision_reason", "—"))

    st.markdown("**Qualification**")

    for f, v in state["qualification"].items():
        st.markdown(
            f"- {f}: {'`' + v + '`' if v else '_unknown_'}"
        )

    st.markdown("**Handoff**")
    st.write(
        "Triggered"
        if state.get("handoff_ready")
        else "Not triggered"
    )

    st.markdown("**Nurture**")

    n = state["nurture"]

    st.markdown(
        f"- attempts: {n['attempt_count']} · "
        f"next contact day: "
        f"{n.get('next_contact_at') or '—'} · "
        f"stopped: {n['stopped']}"
    )

    st.markdown("**Safety**")

    safety_flags = state.get("safety_flags", [])

    if safety_flags:
        for f in safety_flags:
            st.warning(f)
    else:
        st.write("Passed")

    st.markdown("**Response guardrail**")

    guardrail_flags = state.get("guardrail_flags", [])

    if guardrail_flags:
        for f in guardrail_flags:
            st.warning(f)
    else:
        st.write("Passed")

    st.markdown("**Reasoning log (last turn)**")

    for line in state.get("reasoning_log", [])[-10:]:
        st.caption(line)