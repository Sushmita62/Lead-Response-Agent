# Lead Response Agent — LangGraph edition

## Fixes applied to this build (read this first)

This zip was submitted with several integration breaks between files that
had drifted out of sync during iteration. All were found by **actually
running the app**, not just reading the code, and are fixed here:

1. **`tools.py` didn't match `graph.py`** — `graph.py` imported
   `safe_listings_for_llm` and expected `lookup_listing()` to return
   `{"matches": [...], "flags": [...]}`; the shipped `tools.py` was an
   older version with neither. → rewritten to match.
2. **`guardrails.py` didn't match `graph.py`** — `graph.py` called
   `output_guardrail(..., protected_terms=...)` expecting a dict back;
   the shipped `guardrails.py` didn't accept `protected_terms` and
   returned a tuple. Same issue for `input_safety_check`. → both
   rewritten to return dicts and accept `protected_terms`.
3. **`llm.py`'s `call_structured()` signature didn't match how `graph.py`
   called it** — `graph.py` calls `call_structured(prompt=, schema=,
   mock_fn=)`; the shipped `llm.py` required `system_prompt`, `messages`,
   `tool_name`, `tool_schema`. → `llm.py` rewritten to match `graph.py`'s
   calling convention.
4. **The opener-style learning bandit was never wired into the graph** —
   `learning.py`'s `BANDIT` existed and was imported by `app.py`, but
   `graph.py` never called `BANDIT.choose()` or set `state["opener_style"]`,
   so live learning could never update (only the seeded historical prior
   ever showed). → `reply_node` now chooses an opener style via the bandit
   on a lead's first turn and threads it into the reply prompt.
5. **`abuse_count` and `opener_outcome_recorded` were used but never
   declared in the `LeadState` TypedDict** — LangGraph builds its state
   channels from the schema, so any key not declared there is silently
   dropped between node hops. This meant the documented "second abusive
   message disengages" policy could never actually trigger. → both fields
   added to `state.py`.
6. **Save-ordering bug in `app.py`** — `record_outcome_from_turn()` sets
   `opener_outcome_recorded` on the state *after* `graph.py`'s internal
   `save_node` had already persisted it, so the flag never actually made
   it to SQLite and the same opener could be double-scored. → `app.py`
   now re-saves state after learning runs.
7. **Mock fallbacks were message-blind** — `decision_node`'s and
   `understand_intent_node`'s mock functions always returned `reply`/
   `stop_request: False` regardless of what the lead typed, so without a
   live API key the `stop`/`handoff`/`nurture` branches were untestable.
   → both now use a small keyword heuristic so the whole graph is
   exercisable offline.
8. **`requirements.txt` was missing `python-dotenv`**, which `llm.py`
   imports. → added.

**Verified after fixing**: graph compiles, all 8 sample leads process
without error, the injection/PII lead raises the right flags, an explicit
opt-out deterministically routes to `stop`, two consecutive abusive
messages correctly trigger the hard safety block, a handoff phrase
correctly routes to `handoff`, and the Streamlit app was actually launched
and returned HTTP 200 (not just import-checked).

## Handoff behavior fix (post-live-testing)

Live testing surfaced a real issue: `handoff_node` had a **hardcoded**
reply ("I'll pass this along to the realtor...") and `app.py` disabled the
chat input entirely once `handoff_ready` was set, showing "This lead has
been handed off" and ending the conversation. That both violated "no
hardcoded wording" and misrepresented what handoff actually means --
handoff means the realtor follows up separately, not that they've joined
the current thread or that the conversation is over.

Fixed:
- `handoff_node` now calls the LLM (same pattern as `reply_node`) with a
  prompt that explicitly forbids implying the realtor is joining this
  chat or that the conversation is ending, and asks for a fresh,
  context-aware message referencing what the lead actually said.
- `app.py` now shows an informational banner on handoff but keeps
  `st.chat_input` active, so the lead (and evaluator) can keep messaging
  after handoff instead of being cut off.
- `handoff_ready=True` is still set exactly as before -- only the
  reply text and the chat-ending behavior changed, per the minimal-fix
  request.


---

Same agent, rebuilt as a LangGraph `StateGraph` with deterministic tools and
guardrails around three isolated LLM calls, plus SQLite persistence and a
Streamlit front door.

## Quick start

```bash
pip install -r lead_agent/requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...      # or:
export LLM_PROVIDER=groq
export GROQ_API_KEY=gsk_...
cd lead_agent
streamlit run app.py
```

Without any key set, the app runs in a labeled mock mode (visible in every
reasoning-log line) so it's demoable offline.

## Architecture

```
lead_agent/
  state.py        LeadState TypedDict — the one object every node reads/writes
  data_loader.py   loads the 6 provided files, flags data-quality issues
  tools.py         deterministic property/market lookups — no LLM
  guardrails.py    deterministic input + output guardrails — no LLM
  nurture.py       retry cadence math — no LLM
  learning.py      epsilon-greedy bandit over opener style, seeded from
                    past_conversations.json
  llm.py           provider abstraction: LLM_PROVIDER=anthropic|groq,
                    mock fallback if no key is set
  graph.py          the LangGraph itself — 12 nodes, wired below
  storage.py        SQLite persistence (Mira's memory across turns)
  app.py             Streamlit UI: chat + evaluator sidebar
```

## The graph

```
load_state → safety_check ──blocked──────────────┐
                  │ ok                            │
                  ▼                                │
           understand_intent                       │
             │            │                        │
        need_data      no_data                     │
             ▼               │                     │
        property_tool        │                     │
             └──────┬────────┘                     │
                     ▼                              │
              update_state                          │
                     ▼                              │
              decision_agent ◄──────────────────────┘
                     │
        ┌────────┬────────┬────────┐
        ▼        ▼        ▼        ▼
      reply   handoff  nurture   stop
        └────────┴────────┴────────┘
                     ▼
            response_guardrail
                     ▼
                save_state
```

Only three nodes ever call an LLM: `understand_intent`, `decision_agent`,
and the reply-drafting call shared by `reply`/`handoff`/`nurture` (`stop` is
fully deterministic — compliance-critical, never left to a model). Every
other node is plain Python, so nothing a lead types can change what the
safety check, the property lookup, the output guardrail, or the nurture
scheduler actually do — only what the LLM nodes *say*.

## Deterministic guardrails (guardrails.py)

- **Input**: regex-based detection of prompt-injection phrasing, requests
  for unpublished PII (realtor's personal number), Fair-Housing-steering
  language, and abusive language (second offense hard-stops the turn
  before any LLM call).
- **Output**: blocks any phone-number-shaped string in the drafted reply
  that isn't the brokerage's published number, and blocks
  financial-/legal-advice phrasing. A block swaps in a fixed safe reply —
  never the model's own text.
- **Decision overrides**: explicit opt-out and safety-block both force
  `decision = stop` deterministically, bypassing the decision-agent LLM
  call entirely for those cases.

## LLM provider swap

`llm.py` exposes one function, `call_structured()`, used identically by
all three LLM nodes. Switching providers is one env var — no changes to
`graph.py`. Anthropic uses native tool-calling; Groq uses its
OpenAI-compatible endpoint with the `openai` SDK pointed at
`https://api.groq.com/openai/v1`.

Suggested split if you want to optimize for Groq's speed:
- `understand_intent` and `decision_agent` (categorical, lower-stakes) →
  a fast small model (`llama-3.1-8b-instant`)
- the reply-drafting call (highest stakes — tone, hard-limit adherence) →
  the largest available model (`llama-3.3-70b-versatile`)

Set per-node models by editing the `GROQ_MODEL` env var per node call in
`graph.py`, or extend `llm.py` to accept a model override argument.

## Verified

Tested end-to-end in mock mode: graph compiles, all 8 sample leads process
without error through their full node path, the injection/PII lead
(`L-1004`) correctly raises `injection_attempt` + `pii_request` flags and
the reply prompt is instructed not to comply, the nurture-signaling lead
(`L-1008`) correctly routes to `nurture`, and an explicit opt-out message
deterministically routes to `stop` with a fixed reply — bypassing the
decision-agent LLM call entirely.

## Fixes from live-testing round 2

1. **Crash on "+1d/+7d/+30d" (simulated clock)** —
   `nurture.is_due()` used `nurture.get("next_contact_at", sim_day + 1)`.
   A `dict.get()` default only applies when the key is *missing*, not
   when it's present with value `None` -- which is exactly the normal
   state for any lead that hasn't entered nurture yet. Comparing
   `sim_day >= None` raised `TypeError`. → explicit `is None` check added.
2. **Agent repeated the full property address in every single reply**
   (e.g. "1234 Alhambra Circle in Coral Gables" restated verbatim turn
   after turn) — the prompts in `reply_node` and `handoff_node` never
   told the model to avoid restating already-established specifics.
   → added explicit instructions to refer to it naturally ("the
   property") once already stated, and to vary phrasing turn to turn.
3. **Chat UI showed the lead's message and the agent's reply only
   together, after the LLM call finished** -- the lead's own message
   didn't appear until the (slower) agent reply was also ready, which
   reads as unresponsive. → `app.py` now renders the lead's message
   immediately on submit, shows a spinner while the agent drafts its
   reply, then syncs to the persisted final state.

## Fix: simulated-clock nurture ticks didn't actually work

Root cause: the `+1d/+7d/+30d` sweep in `app.py` called the *full* graph
(`run_turn`) with a placeholder string (`"[no reply — automated nurture
check-in]"`) standing in for the lead's message. But every node in the
pipeline treats `state["message"]` as something the lead actually said --
`update_state_node` even appended it to `history` as a fake `role: "lead"`
turn. Worse, `decision_agent` had no real signal in that placeholder to
re-derive `decision = "nurture"`, so the scheduled follow-up could silently
misfire as something else, and `nurture_node` (the only place that
advances `next_contact_at`) was never reliably reached again -- so after
the first tick, `is_due()` could just keep returning `True` forever without
correctly rescheduling.

Fixed: added `run_nurture_followup()` in `graph.py`, a standalone function
used only by the clock sweep. It does the scheduling math directly
(`attempt_count`, `next_contact_at`, `should_stop`) and drafts one
check-in message via the LLM, without touching `understand_intent` or
`decision_agent`. `app.py`'s sweep now calls this instead of `run_turn`,
and only appends a real `agent` turn to history -- no fake lead message.

Verified: a lead in nurture state, advanced to its scheduled day, now
correctly receives one follow-up message and reschedules to the next
cadence day (tested attempt 1 → 2, day 1 → 4, matching `CADENCE_DAYS`).

## Fix: handoff explanation repeating on every subsequent turn

After a lead was handed off, follow-up questions (e.g. "which time is
best to reach the realtor?") re-triggered the full handoff explanation
again instead of just answering the question -- because nothing in
`decision_node` distinguished "just decided to hand off this turn" from
"already handed off, lead is now asking a normal follow-up."

Fixed: `decision_node` in `graph.py` now checks `state.get("handoff_ready")`
right after the existing stop-request check. If a lead has already been
handed off, decision deterministically routes to `"reply"` instead of
calling the LLM to re-decide -- `handoff_ready` stays `True` throughout,
it's just no longer treated as a reason to repeat the handoff message.
An explicit stop request is still checked first, so a lead can still opt
out after being handed off. `reply_node` already had `realtor_facts` and
full conversation history in its prompt, so no change was needed there --
it can answer post-handoff questions naturally on its own.

Verified: reproduced the exact scenario (handoff on turn 1, two follow-up
questions after) -- `decision` correctly stayed `"reply"` on turns 2 and 3
while `handoff_ready` remained `True` throughout. `py_compile` passed on
all files, Streamlit app confirmed to start (HTTP 200).

## Zero-tolerance hallucination fixes (major round)

Live testing across 8 sample-lead conversations surfaced severe fabrication:
invented Orlando/Naples/Sarasota neighborhoods and listings (dataset only
covers Coral Gables/Coconut Grove/Pinecrest), false claims of sent emails/
texts/bookings, invented mortgage rates and down-payment figures, Fair
Housing steering by neighborhood-for-families framing (English and
Spanish), and a wrong business-hours claim despite the real figure existing
in `brokerage_config.json`.

### Root causes found by inspection (not guessed)
1. `property_lookup_node` only supported single property_id lookup -- no
   broad area/beds/budget search existed, so any question without a
   specific listing ID returned a completely empty `tool_results`, which
   the LLM was free to fill from its own world knowledge.
2. Nothing checked whether an area was even in the realtor's actual
   `service_areas` before the LLM started describing it.
3. `understand_intent_node`'s schema had no `address_hint` field at all --
   address-based lookups beyond the very first turn could never resolve.
4. `realtor_and_brokerage_facts()` never exposed `business_hours_default`,
   even though it's real data in `brokerage_config.json`.
5. `reply_node`'s prompt had zero instruction against claiming completed
   external actions (send/text/email/book) -- the system has no such
   integrations at all.
6. `FAIR_HOUSING_PATTERNS` was too narrow -- missed "safest neighborhoods
   for families / best schools" framing entirely, in both English and
   Spanish, confirmed failing live.
7. `output_guardrail()` had no numeric property-grounding check at all.
8. A pre-existing "first-turn fallback" in `understand_intent_node` always
   re-anchored to the lead's *original* inquiry property whenever nothing
   else resolved -- even when the lead asked about a different or
   nonexistent address, silently returning the wrong (real) listing
   instead of correctly reporting "not found."
9. `reply_node`'s mock fallback is a static string with no visibility --
   a live-call failure (e.g. rate limit) partway through a long
   conversation silently produced the same canned line on every
   subsequent turn with no diagnostic trace.

### Fixes (deterministic, not prompt-only -- per explicit requirement)
- `tools.py`: added `is_area_served()` and `search_listings()` (grounded
  filtering over the real dataset), exposed `business_hours_default`.
- `guardrails.py`: broadened Fair Housing patterns (EN/ES) and added a
  Spanish output-side backstop; added `EXTERNAL_ACTION_CLAIM_PATTERNS`;
  broadened `FINANCIAL_ADVICE_PATTERNS` to catch invented rates/down
  payments; added `contains_ungrounded_numeric_claim()` -- extracts every
  $ amount / sq ft figure from a drafted reply and blocks any that isn't
  traceable to actual retrieved listing data or conversation history;
  fixed a comma-formatting bug in that check during testing (caught before
  shipping); broadened `INJECTION_PATTERNS` ("ignore your..." wasn't
  matched); added owner-PII patterns (previously only realtor-PII was
  covered, not property-owner PII).
- `graph.py`: added `address_hint` extraction; rewrote `property_lookup_node`
  to chain property_id -> address_hint -> area-served check -> broad
  search, with explicit grounding flags surfaced into `reply_node`'s
  prompt; fixed the first-turn-fallback bug (now only applies when no
  address_hint was given, not on every unresolved lookup); added a
  deterministic Fair Housing bypass in `decision_node` that skips the LLM
  entirely for flagged input (prompt-only instruction was demonstrated
  live to not be reliable enough); added explicit "never claim external
  actions" and "never invent financial figures" rules to `reply_node` and
  `handoff_node`; wired the numeric-grounding check into
  `response_guardrail_node`; surfaced `_mode` into the reasoning log so a
  silent mock fallback is now diagnosable.
- `state.py`: declared `fair_housing_redirect` in the `LeadState` schema --
  found and fixed a second instance of the earlier `abuse_count`-class bug
  (LangGraph silently drops undeclared keys between nodes), which meant
  the new Fair Housing bypass was being set but never actually reaching
  `reply_node` until this was fixed.
- `data_loader.py`: added typo detection for city names (found and flagged
  a real data-quality issue -- `L-020`'s city is "Corla Gables", a typo of
  "Coral Gables" -- which would otherwise have caused a false "area not
  served" result for that listing).
- Improved offline mock heuristics (`_mock_understand`) to extract
  addresses, area, and budget so the above fixes are actually testable
  without a live API key -- these mock improvements don't affect live
  behavior, only offline demoability.

### Verified (12/12 test cases pass, run directly against the code)
Orlando/unserved-area correctly flagged rather than described; nonexistent
address correctly returns "not found" instead of the wrong real listing;
Fair Housing redirect fires deterministically in English and Spanish;
owner-PII request correctly flagged on input and blocked on output if
leaked; prompt injection ("ignore your previous instructions...") now
detected; mid-conversation area/budget change correctly overrides stale
qualification state; false external-action claims blocked; invented
mortgage rates blocked; stale/sold-in-notes listings confirmed flagged;
fabricated listing details blocked by the numeric grounding check.
`py_compile` passes on all 10 files; Streamlit app confirmed to start
(HTTP 200) with every change applied together.

### Residual risks NOT fully eliminated (reporting honestly, as required)
- **Numeric grounding is a heuristic, not a parser.** It catches invented
  $ amounts and sq ft figures, but cannot verify prose claims ("recently
  built," "walking distance to schools," "gated community") that aren't
  numeric. A model could still fabricate qualitative property descriptions
  that pass this check.
- **`contains_ungrounded_numeric_claim` could theoretically false-positive**
  if two unrelated numbers in the grounded-facts string happen to
  concatenate (after whitespace/comma stripping) into a substring matching
  a fabricated number. Not observed in testing, but not structurally
  impossible at this dataset's scale.
- **`search_listings()`'s city matching is exact (case-insensitive), not
  fuzzy** -- it will not find `L-020` (Coral Gables, typo'd as "Corla
  Gables") when searching for "Coral Gables," even though the typo is now
  flagged as a data-quality issue. The flag exists; the search doesn't
  compensate for it.
- **The Fair Housing pattern list is still pattern-based, not semantic.**
  Novel phrasings not resembling the patterns tested here (e.g. an
  indirect dog-whistle with no "family/school/safe" keywords at all) could
  still slip through the deterministic check and reach the LLM's judgment
  alone, which was already demonstrated to be an insufficient sole
  safeguard.
- **External-action-claim patterns are also pattern-based.** A sufficiently
  different phrasing of a false completed-action claim (not matching any
  of the current regexes) could still pass through undetected.
- **No live-mode verification was possible from the development
  environment** (no network access to the Groq API from this sandbox) --
  every test above was run against mock mode and direct function calls,
  which exercises the same deterministic guardrail code paths but not the
  live LLM's actual prompt-following behavior. Recommend re-running the
  12 test cases with a live key before considering this fully verified.

## Fix: multi-turn property continuity broken (over-correction from prior round)

Live testing (real Groq calls) surfaced a regression I introduced in the
previous round: after fixing the "wrong property re-anchored" bug, a lead
asking a vague follow-up ("tell me more about this property", "how old is
it") on turn 2+ got "I don't have that information" even though the exact
same property's full data (including year_built and days_on_market) was
sitting right there in the dataset.

Root cause: the fix from last round only restored context on the literal
first turn (`_is_first_agent_turn(state)`). Turn 2 onward, if the lead
used a pronoun instead of restating the address, neither `property_id`
nor `address_hint` resolved, and nothing else remembered which property
was being discussed -- so `tool_results` came back genuinely empty.

Fix: replaced the first-turn-only hack with `active_listing_id`, a proper
piece of persistent state (declared in `state.py`) that:
- is seeded from the lead's original inquiry on load (`load_state_node`)
- updates whenever a `property_id` or `address_hint` resolves to exactly
  one real listing (`property_lookup_node`)
- is used as the fallback lookup whenever neither is given on a later
  turn -- so "this property" / "it" correctly continues referring to
  whatever was most recently established, not just the very first one

Verified: reproduced the exact scenario (opening inquiry -> vague
follow-up) -- `active_listing_id` and the resolved listing now persist
correctly. Also verified no regression on the nonexistent-address case
from the prior round (still correctly returns "not found," doesn't fall
back to the wrong property).

## Fix: grounding rules were over-strict, blocking answerable questions

The strict grounding language added in the previous round explicitly
named certain fields as safe to state (price, beds, baths, sqft, year
built, taxes, HOA, availability) but never mentioned `days_on_market` or
`listing_agent` -- and the "never estimate" wording appears to have made
the model refuse simple arithmetic (property age from `year_built`) even
though the underlying fact was fully grounded.

Fixed: `days_on_market`, `subdivision`, and `listing_agent` added to the
explicitly-named allowed fields; added a rule explicitly permitting
arithmetic ON grounded figures ("if year_built is given, you may state
the property's age using today's date... this is calculation from real
data, not estimation"); added today's actual date to the prompt as a
concrete anchor for age calculations.

## Fix: false-promise patterns broadened after live-testing gaps

Live testing showed two phrasings that slipped past the guardrails added
last round:
- "I'll have a trusted mortgage specialist reach out to you shortly" --
  the existing pattern only covered "X will reach out," not the causative
  "I'll have X reach out" construction.
- "they'll reach out to you shortly" -- the existing pattern only covered
  named subjects (she/he/Rebecca/the realtor), not the generic pronoun.

Both patterns added to `EXTERNAL_ACTION_CLAIM_PATTERNS`. Deliberately kept
the urgency qualifier (shortly/soon/right away/today) as a requirement --
a plain "I'll have Rebecca follow up with you" (no timing promise) is
legitimate handoff framing per the case study's own guidance and
shouldn't be over-blocked; verified both the failure case blocks and the
benign case doesn't.

## Added: realtor-level data-quality flags now surfaced to the agent

`DB["realtor_flags"]` (e.g. `service_area_outlier:Austin` -- a genuine
anomaly in the provided data, since Austin, TX doesn't fit the realtor's
otherwise Miami-area service footprint) was computed at load time but
never actually reached the LLM's prompt. Now included in `reply_node`'s
context so the agent can naturally flag it when relevant (e.g. when
discussing the one Austin listing) instead of treating it as an entirely
unremarkable data point.

## Fix: handoff banner now shows once, not on every subsequent turn

`app.py` previously re-rendered the "passed to the human realtor" banner
on every single turn for the rest of the conversation, since it was keyed
purely off `handoff_ready` staying `True`. Now tracked with a
session-scoped "already seen" set per lead_id, so it displays once, right
when handoff occurs, and not again afterward.

## Verified live-mode accuracy (real Groq calls, from the user's own testing)

Two properties quoted in a live conversation (4520 Coral Ridge Dr,
Pinecrest and 1100 W 6th St, Austin -- including price, beds, baths,
sqft, year built, tax, HOA, and listing agent) were checked directly
against `mock_listings.csv` and matched exactly. The grounding fixes from
the previous round are holding up under real LLM calls, not just mock
testing.
