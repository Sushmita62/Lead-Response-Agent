"""
Deterministic tools used by the Lead Response Agent.
 
These functions do not call an LLM. They only retrieve and prepare
trusted case-study data for the graph.
 
Important:
- Never expose owner_name / owner_phone to lead-facing generation.
- Preserve messy source data and its quality flags.
- When an address has multiple listings, return all matches.
"""
 
from data_loader import DB
 
 
def lookup_listing(property_id: str = None, address_hint: str = None):
    """
    Look up listings by property ID or address.
 
    Returns:
        {"matches": [...], "flags": [...]}
 
    We return all address matches because the case study deliberately
    contains duplicate addresses with different prices.
    """
    if property_id:
        listing = DB["listing_by_id"].get(property_id)
 
        if listing:
            return {
                "matches": [listing],
                "flags": list(listing.get("flags", [])),
            }
 
        return {
            "matches": [],
            "flags": ["referenced_listing_not_found"],
        }
 
    if address_hint:
        hint = str(address_hint).strip().lower()
 
        matches = [
            listing
            for listing in DB["listings"]
            if hint in str(listing.get("address", "")).lower()
        ]
 
        if not matches:
            return {"matches": [], "flags": ["no_address_match"]}
 
        flags = []
        for listing in matches:
            flags.extend(listing.get("flags", []))
 
        if len(matches) > 1:
            flags.append("multiple_address_matches")
 
        # de-dupe while preserving order
        flags = list(dict.fromkeys(flags))
 
        return {"matches": matches, "flags": flags}
 
    return {"matches": [], "flags": []}
 
 
def lookup_market(area: str):
    """Returns None if the area is not present in the provided data."""
    if not area:
        return None
 
    areas = DB.get("market", {}).get("areas", {})
 
    if area in areas:
        return areas[area]
 
    target = str(area).strip().lower()
    for name, snapshot in areas.items():
        if str(name).strip().lower() == target:
            return snapshot
 
    return None
 
 
def is_area_served(area: str) -> bool:
    """True only if the realtor's actual declared service_areas include this
    area. Used to stop the agent discussing markets (e.g. Orlando, Naples)
    that simply are not in the dataset at all -- rather than letting the
    LLM freelance from its own world knowledge when tool_results is empty."""
    if not area:
        return True  # nothing to check yet -- not a false claim either way
    served = [a.strip().lower() for a in DB["realtor"].get("service_areas", [])]
    return str(area).strip().lower() in served
 
 
def search_listings(area: str = None, beds_min=None, price_max=None, limit: int = 5):
    """Deterministic filtered search over the actual listing dataset. This
    exists so a broad question ("5-bedroom homes in X under $Y") is answered
    from real matching rows, not left for the LLM to invent when a single
    lookup_listing() call finds nothing."""
    results = []
    for listing in DB["listings"]:
        if area and str(listing.get("city", "")).strip().lower() != str(area).strip().lower():
            continue
        try:
            beds = float(listing.get("beds") or 0)
        except (TypeError, ValueError):
            beds = 0
        if beds_min is not None and beds < float(beds_min):
            continue
        price = listing.get("price")
        if price_max is not None and (price is None or price > float(price_max)):
            continue
        results.append(listing)
        if len(results) >= limit:
            break
    return results
 
 
def realtor_and_brokerage_facts():
    """Only information safe for lead-facing generation. Internal/private
    brokerage fields are intentionally excluded."""
    realtor = DB["realtor"]
    brokerage = DB["brokerage"]
 
    return {
        "realtor_name": realtor.get("name"),
        "years_experience": realtor.get("years_experience"),
        "specialties": realtor.get("specialties"),
        "service_areas": realtor.get("service_areas"),
        "languages_spoken": realtor.get("languages_spoken"),
        "tone_preferences": realtor.get("tone_preferences"),
        "brokerage_legal_name": brokerage.get("legal_name"),
        "assistant_name": brokerage.get("assistant_name"),
        "published_phone": brokerage.get("published_phone", realtor.get("phone", "")),
        "published_email": brokerage.get("published_email", realtor.get("email", "")),
        "business_hours_brokerage": brokerage.get("business_hours_default", ""),
        "working_hours_realtor": realtor.get("working_hours", ""),
        # recent_transactions: kept exactly as-is, including any flagged
        # future-dated entry -- data_loader.py already flags that in
        # DB["realtor_flags"], which reaches the prompt separately, so the
        # agent can be transparent about it rather than silently omitting
        # or silently trusting a bad date.
        "recent_transactions": realtor.get("recent_transactions", []),
    }
 
 
BLOCKED_LISTING_FIELDS = {"owner_name", "owner_phone"}
 
 
def safe_listing_for_llm(listing: dict):
    """Strip owner/private fields before a listing reaches an LLM prompt.
    Architectural protection, not just a prompt instruction."""
    if not listing:
        return {}
    return {k: v for k, v in listing.items() if k not in BLOCKED_LISTING_FIELDS}
 
 
def safe_listings_for_llm(listings):
    return [safe_listing_for_llm(l) for l in (listings or [])]
 