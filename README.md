# Lead Response Agent

An AI agent that responds to inbound real-estate leads over chat, qualifies
them conversationally, looks up real property/market data, hands off to a
human realtor when appropriate (with an email notification), and keeps
following up over time — built as a LangGraph state machine with
deterministic guardrails around every LLM call.

## Quick start

```bash
cd lead_agent
pip install -r requirements.txt
cp .env.example .env   # then fill in your real keys
streamlit run app.py
```

Required in `.env`:
- `GROQ_API_KEY` / `GROQ_MODEL` — the conversational LLM (Groq-hosted)
- `RESEND_API_KEY` — sends the handoff notification email (optional; the
  app runs fine without it, handoff just won't email anyone)

## How a message flows through the system

Every incoming lead message runs through one pass of the graph below.
Nothing in this pipeline is skipped — the same 12 steps run whether it's
the lead's first message or their fifteenth.

```
Lead message
     │
     ▼
┌─────────────┐
│ load_state  │  loads/initializes conversation state, seeds qualification
│             │  from the lead's saved_search, seeds active_listing_id
└─────┬───────┘  from their original inquiry
      ▼
┌─────────────┐
│ safety_check│  regex scan for prompt injection, PII requests, Fair
│             │  Housing steering language, abuse (2nd offense = hard stop)
└─────┬───────┘
      │
      ├── flagged for hard stop ──────────────────────┐
      │                                                ▼
      ▼                                       ┌─────────────────┐
┌─────────────┐                                │   safe_stop     │
│  understand │  LLM extracts intent, property  │ (no LLM call,   │
│             │  reference, language, timeline/ │  fixed reply)   │
│             │  budget/area/financing/etc.,     └────────┬────────┘
│             │  stop-request flag                        │
└─────┬───────┘                                            │
      ▼                                                     │
┌─────────────┐                                             │
│ property_   │  resolves the property in focus:            │
│  lookup     │  exact ID → address text → "still talking    │
│             │  about the same one" fallback. If the        │
│             │  resolved listing shares an address with     │
│             │  another (duplicate MLS records), fetches    │
│             │  ALL of them. Also runs a broad area/beds/    │
│             │  budget search across the full dataset,       │
│             │  independent of whether a specific listing    │
│             │  also resolved -- so "show me another one"    │
│             │  always has real data to work from. Pulls     │
│             │  the matching market snapshot too.            │
└─────┬───────┘                                              │
      ▼                                                       │
┌─────────────┐                                               │
│ update_state│  merges newly-extracted qualification fields   │
│             │  into persistent state -- latest explicit      │
│             │  statement always overrides stale data         │
└─────┬───────┘                                                │
      ▼                                                         │
┌─────────────┐                                                 │
│  decision   │  LLM decides reply / handoff / nurture / stop,   │
│             │  but three cases are decided deterministically   │
│             │  BEFORE the LLM is even asked: explicit opt-out   │
│             │  → stop; Fair Housing steering detected → reply    │
│             │  (with a fixed redirect, no LLM call); already      │
│             │  handed off earlier this conversation → reply       │
│             │  (don't re-explain handoff every turn). Otherwise    │
│             │  the LLM decides, seeing both the specific listing    │
│             │  AND the broader search results, so "suggest a         │
│             │  property" isn't mistaken for "needs a human."          │
└─────┬───────┘                                                          │
      │                                                                   │
      ├──────────────┬──────────────┬──────────────┐                     │
      ▼              ▼              ▼              ▼                     │
  ┌───────┐     ┌──────────┐   ┌─────────┐    ┌────────┐                 │
  │ reply │     │ handoff  │   │ nurture │    │  stop  │                 │
  └───┬───┘     └────┬─────┘   └────┬────┘    └───┬────┘                 │
      │              │              │             │                      │
      │   LLM drafts │  sends       │  LLM drafts │  fixed reply,        │
      │   a grounded  │  handoff     │  a low-      │  no LLM call        │
      │   reply, only  │  email      │  pressure     │                    │
      │   describing    │  (Resend),  │  check-in,     │                    │
      │   real listing   │  schedules  │  reschedules    │                    │
      │   data actually   │  a follow-up│  next contact     │                    │
      │   retrieved above  │  if it     │                     │                    │
      │                     │  fails,    │                     │                    │
      │                     │  the       │                     │                    │
      │                     │  turn just │                     │                    │
      │                     │  continues │                     │                    │
      ▼                     ▼            ▼                     ▼                    │
      └──────────────┬──────────────────┴─────────────────────┘                    │
                      ▼                                                             │
            ┌───────────────────┐                                                   │
            │ response_guardrail│◄──────────────────────────────────────────────────┘
            │                   │  three independent checks on the drafted reply:
            │                   │  1) regex: unpublished phone numbers, financial-
            │                   │     advice language, external-action-claim phrasing
            │                   │  2) numeric grounding: every $ amount / sq ft figure
            │                   │     in the reply must trace back to real retrieved
            │                   │     data, or it's blocked
            │                   │  3) semantic (LLM-judged, any language): Fair
            │                   │     Housing steering, false action claims, financial
            │                   │     advice -- catches what regex patterns miss
            │                   │  any failure swaps in a safe fallback reply
            │                   │  (language-matched, no unfounded promises)
            └─────────┬─────────┘
                      ▼
                ┌───────────┐
                │   save    │  persists state to SQLite, appends to history
                └─────┬─────┘
                      ▼
                Reply shown to lead
```

Two things happen **outside** this per-message flow:

- **Automated nurture check-ins.** The Streamlit sidebar's simulated clock
  (`+1d`/`+7d`/`+30d`) checks every lead's `nurture.next_contact_at`, and
  fires a standalone follow-up message for anyone due — this bypasses the
  full graph (there's no real new lead message to run intent/decision on),
  going straight to a scheduling calculation plus one LLM-drafted check-in.
- **Learning.** After each reply, an epsilon-greedy bandit records whether
  the opener style used (direct/warm/question-first) led to a lead
  reply — seeded at startup from the 6 provided past conversations, and
  visible live in the sidebar.

## File map

```
lead_agent/
  app.py            Streamlit UI -- chat pane + evaluator sidebar
  graph.py           the LangGraph itself: all nodes and edges above
  state.py           the shared state schema every node reads/writes
  llm.py             Groq API wrapper -- structured tool-calling, mock
                      fallback when no key is set
  tools.py           deterministic lookups: single listing, broad search,
                      market snapshot, service-area check, realtor/
                      brokerage facts (privacy-filtered)
  guardrails.py       regex-based input/output safety checks
  data_loader.py      loads the 6 data files, flags data-quality issues
                      (duplicate addresses, malformed prices/dates,
                      missing fields, city-name typos, business-hours
                      contradictions) without silently fixing them
  learning.py         the opener-style bandit
  nurture.py          retry cadence math (1/3/7/21/30/30 day schedule)
  storage.py           SQLite persistence
  email_service.py     Resend integration for handoff notifications
  debug_logger.py      structured console + file logging for every node/
                        tool call, used across the whole graph
data/
  realtor_profile.json, brokerage_config.json, mock_listings.csv,
  market_snapshots.json, sample_leads.json, past_conversations.json
  -- the 6 source-of-truth files, loaded once at startup, never modified
```

## What each capability actually does

**Property lookup.** Never just returns one listing and calls it done.
Checks for duplicate MLS records at the same address and surfaces all of
them (so the agent can ask "which one do you mean" instead of guessing).
Checks whether the requested area is even in the realtor's real service
footprint before describing it. Runs a broad dataset search whenever
there's enough criteria to search on, independent of whether a specific
listing also happens to be in focus.

**Handoff.** Triggered by the decision step, not gated behind collecting
every qualification field — a lead with just a clear area+budget and a
request for human help is enough. Sends one email via Resend containing
lead contact info (pulled from the lead record, never re-asked), the
qualification gathered so far, the specific property if one's known (or a
clear "no specific listing — see preferences" if not), and every duplicate
listing at that address if there are more than one. Never repeats the
handoff explanation on later turns — the conversation continues normally,
with the human simply now also in the loop.

**Nurture.** For leads who aren't ready, not leads who've gone silent for
no reason — an explicit "not ready yet" signal schedules a future,
escalating-interval check-in (1, 3, 7, 21, 30, 30 days) rather than either
nagging or dropping them.

**Guardrails.** Layered, not single-point: regex catches known bad
patterns fast and cheap; a numeric-grounding check independently verifies
every number in a reply against real retrieved data; a semantic LLM check
catches what regex can't, including non-English phrasing. All three run on
every reply, not just handoff/reply — nurture and stop messages go through
the same check.

**Language matching.** The lead's detected language flows through the
whole turn — extraction, decision, reply generation, and even the
guardrail fallback messages (English/Spanish variants), so a blocked reply
mid-Spanish-conversation doesn't suddenly switch to English.