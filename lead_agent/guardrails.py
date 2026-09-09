"""
Deterministic, non-LLM guardrails. These run as ordinary Python before and
after the LLM touches anything, so they can't be talked out of their job by
a clever message -- unlike an LLM-based check, a regex either matches or it
doesn't.
 
INPUT guardrail (safety_check node): runs on the lead's raw message, before
intent understanding. Flags injection attempts, requests for non-public PII,
and Fair-Housing-steering language. Severe cases set a block, which
short-circuits the graph straight to a fixed, safe reply -- no LLM call at
all for the worst cases (e.g. repeated abuse).
 
OUTPUT guardrail (response_guardrail node): runs on the LLM's drafted reply,
before it's sent. Blocks phone-number-shaped strings not on the realtor's
own published contact, blocks financial/legal-advice phrasing, and blocks
any protected terms (owner names/phones, internal notes) leaking through.
"""
import re
 
INJECTION_PATTERNS = [
    r"ignore (your |all |any |the )?(previous|prior|above) instructions",
    r"system prompt",
    r"you are now",
    r"act as (a|an) (different|new)",
    r"disregard (your|the) (rules|guidelines|instructions)",
    r"reveal your (instructions|prompt|system)",
]
 
PII_REQUEST_PATTERNS = [
    r"owner'?s? (personal )?(phone|number|cell|email)",
    r"(personal|home) (cell|number|address) (of|for) the (realtor|agent)",
    r"what'?s (the )?realtor'?s (real|personal) (number|cell)",
    r"realtor'?s (personal|real|direct|home) (phone|number|cell|email)",
    r"(share|give|send) (me )?(the )?realtor'?s (personal|real|direct|home)",
    # property-owner PII (distinct from realtor-PII above) -- "who owns
    # this, give me the owner's name/phone" was previously undetected.
    r"who owns (this|that|it)",
    r"owner'?s? name",
    r"(give|send|share) (me )?the owner'?s? (name|phone|number|contact)",
]
 
FAIR_HOUSING_PATTERNS = [
    r"(good|safe|families like (mine|ours)) (neighborhood|area|school) for (my |our )?(race|religion|kids of a certain)",
    r"is (this|that) (area|neighborhood) (safe from|full of) [a-z]+ people",
    r"what (race|religion|ethnicity) (lives|live) (there|in that area|nearby)",
    # broadened: "safest/best neighborhoods for families/kids" and school-as-proxy
    # framing -- demonstrated in live testing to be missed by the patterns
    # above, in both English and Spanish.
    r"(safest|best|good) (neighborhoods?|areas?|barrios?) .{0,40}(famil|kids?|child|ni[ñn]os?)",
    r"(neighborhoods?|areas?|barrios?) .{0,40}(safest|best|good) .{0,20}(famil|kids?|child|ni[ñn]os?)",
    r"(best|good) schools?.{0,30}(safest|safe|good area|good neighborhood)",
    r"(m[aá]s )?seguro?s? .{0,30}(famil|ni[ñn]os?)",
    r"(barrios?|zonas?) .{0,30}(seguros?|mejores?) .{0,30}(famil|ni[ñn]os?)",
]
 
# Same patterns, applied to the LLM's own drafted reply as a backstop --
# input-side detection alone was demonstrated (live testing) to miss cases
# where the lead's phrasing was borderline but the model still answered
# with neighborhood-by-family/school framing anyway.
OUTPUT_FAIR_HOUSING_PATTERNS = FAIR_HOUSING_PATTERNS + [
    r"(family|families)[\s\-]friendly (area|neighborhood)",
    r"(neighborhoods?|areas?) .{0,60}(families|kids) (tend to|often|usually) (enjoy|love|prefer)",
]
 
EXTERNAL_ACTION_CLAIM_PATTERNS = [
    # false claims of a completed action -- this system has no actual
    # email/SMS/CRM/booking integration, so these can never be true.
    r"i('ve| have| just)? (already )?(sent|texted|emailed|e-mailed) (you|the|it|that)",
    r"i('ve| have| just)? (already )?(notified|informed|told) (rebecca|the realtor)",
    r"i('ve| have| just)? (already )?(booked|scheduled|confirmed|arranged) (a |the )?(showing|viewing|appointment|call)",
    r"(she|he|rebecca|the realtor|they)('ll| will) (call|text|email|reach out to) you (shortly|soon|right away|today)",
    r"(a |the )?(trusted )?(mortgage )?(specialist|lender|agent)('ll| will) (reach out|contact|call|text|email) (you )?(shortly|soon|right away|today)",
    r"they('ll| will) reach out to you (shortly|soon|right away|today)",
    r"i('ll| will) have (a |the )?(trusted )?(mortgage )?(specialist|lender|agent|realtor|rebecca) (reach out|call|text|email|contact)[a-z\s]{0,20}(shortly|soon|right away|today)",
    r"i('ll| will) (monitor|keep an eye on|watch) the market and (alert|notify|let) you",
    r"you should (see|receive) (it|the (email|text|list)) (shortly|soon)",
]
 
ADDRESS_UNAVAILABLE_PATTERNS = []  # reserved
 
ABUSE_PATTERNS = [
    r"\b(fuck|shit|asshole|bitch)\b.{0,20}\b(you|bot|agent)\b",
]
 
PHONE_RE = re.compile(r"(?<!\d)(\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")
 
FINANCIAL_ADVICE_PATTERNS = [
    r"you should (definitely )?(buy|invest|refinance)",
    r"this is (a )?guaranteed (return|investment)",
    r"i (recommend|advise) you (take out|sign|accept) (a |the )?(loan|mortgage|offer)",
    # broadened: specific invented rates/amounts, demonstrated in live
    # testing (e.g. "rates are generally in the 6%-7% range", "down payment
    # ranges from about 3.5%-20%") -- none of this is in the provided data.
    r"\d+(\.\d+)?\s*%.{0,20}(mortgage|interest|rate)",
    r"(mortgage|interest) rates?.{0,30}\d+(\.\d+)?\s*%",
    r"down\s*payment.{0,30}(\$|\d+(\.\d+)?\s*%)",
    r"\$[\d,]+.{0,20}down\s*payment",
]
LEGAL_ADVICE_PATTERNS = [
    r"legally you (can|must|have to)",
    r"that would (violate|be illegal under)",
]
 
# Numeric claims that look like property specifics (price, sqft, bed/bath
# counts). Used by contains_ungrounded_numeric_claim() below to catch
# invented listing details, e.g. a fabricated "$595,000, 2,450 sq ft"
# description for a market the dataset doesn't even cover.
_PRICE_RE = re.compile(r"\$\s?([\d,]{4,})")
_SQFT_RE = re.compile(r"([\d,]{3,})\s*(sq\s?\.?\s?ft|square feet)", re.IGNORECASE)
 
 
def _any_match(patterns, text):
    t = (text or "").lower()
    return [p for p in patterns if re.search(p, t)]
 
 
def input_safety_check(message: str, prior_abuse_count: int = 0):
    """Returns {"flags": [...], "block": str|None}."""
    flags = []
    if _any_match(INJECTION_PATTERNS, message):
        flags.append("injection_attempt")
    if _any_match(PII_REQUEST_PATTERNS, message):
        flags.append("pii_request")
    if _any_match(FAIR_HOUSING_PATTERNS, message):
        flags.append("fair_housing_steering")
    abusive = _any_match(ABUSE_PATTERNS, message)
    if abusive:
        flags.append("abusive_language")
 
    block = None
    if abusive and prior_abuse_count >= 1:
        # second strike -- disengage without another LLM call
        block = "Repeated abusive language after one redirect. Disengaging per policy."
 
    return {"flags": flags, "block": block}
 
 
def output_guardrail(reply_text: str, published_phone: str = "", protected_terms=None):
    """Returns {"flags": [...], "blocked": bool}."""
    flags = []
    blocked = False
    text = reply_text or ""
 
    published_digits = re.sub(r"\D", "", published_phone or "")
    for m in PHONE_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if digits and digits != published_digits and not (
            published_digits and published_digits.endswith(digits[-7:])
        ):
            flags.append("unpublished_phone_number_in_reply")
            blocked = True
 
    if _any_match(FINANCIAL_ADVICE_PATTERNS, text):
        flags.append("financial_advice_language")
        blocked = True
    if _any_match(LEGAL_ADVICE_PATTERNS, text):
        flags.append("legal_advice_language")
        blocked = True
    if _any_match(EXTERNAL_ACTION_CLAIM_PATTERNS, text):
        flags.append("false_external_action_claim")
        blocked = True
    if _any_match(OUTPUT_FAIR_HOUSING_PATTERNS, text):
        flags.append("fair_housing_steering_in_reply")
        blocked = True
 
    for term in (protected_terms or []):
        term = str(term).strip()
        if term and term.lower() in text.lower():
            flags.append("protected_term_leaked")
            blocked = True
 
    return {"flags": flags, "blocked": blocked}
 
 
def contains_ungrounded_numeric_claim(reply_text: str, grounded_text: str) -> bool:
    """
    Deterministic property-grounding check. Extracts every dollar amount and
    every "N sq ft" figure from the drafted reply, and blocks if any of them
    doesn't appear anywhere in the grounded context (the actual listing/
    search data retrieved this turn, plus prior conversation history).
 
    This is a heuristic, not a perfect parser -- it can't verify prose
    claims like "recently built" or "walking distance to the school," only
    specific numbers. But numbers are exactly what got fabricated wholesale
    in testing (e.g. an invented "$595,000, 2,450 sq ft" home in a market
    the dataset doesn't even cover), so this catches the highest-value case.
    """
    reply_text = reply_text or ""
    grounded_text = grounded_text or ""
    grounded_digits_only = re.sub(r"[,\s]", "", grounded_text)
 
    for m in _PRICE_RE.finditer(reply_text):
        amount = m.group(1).replace(",", "")
        if amount not in grounded_digits_only:
            return True
 
    for m in _SQFT_RE.finditer(reply_text):
        amount = m.group(1).replace(",", "")
        if amount not in grounded_digits_only:
            return True
 
    return False
 
 
SAFE_FALLBACK_REPLY = (
    "I want to make sure you get accurate info on this -- let me have "
    "the realtor follow up directly rather than guess."
)
 
SAFE_FALLBACK_REPLY_ES = (
    "Quiero asegurarme de darte información precisa sobre esto -- "
    "dejaré que la agente inmobiliaria te contacte directamente en lugar "
    "de adivinar."
)
 