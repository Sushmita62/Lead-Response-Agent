import os
import resend

resend.api_key = os.getenv("RESEND_API_KEY")


def send_handoff_email(
    lead_name,
    lead_id,
    reason,
    qualification,
    last_message,
    contact=None,
    properties=None,
):
    contact = contact or {}
    properties = properties or []

    property_html = ""

    if properties:
        property_html = "<h3>Interested / Relevant Properties</h3><ul>"

        for p in properties:
            property_html += f"""
            <li>
                <strong>{p.get('listing_id') or p.get('property_id') or 'Unknown ID'}</strong>
                — {p.get('address', 'Address unavailable')}
                — ${p.get('price', 'Price unavailable')}
                — {p.get('beds', 'N/A')} beds / {p.get('baths', 'N/A')} baths
            </li>
            """

        property_html += "</ul>"

    params = {
        "from": "onboarding@resend.dev",
        "to": ["sushmitaraj6202@gmail.com"],
        "subject": f"New Lead Handoff — {lead_name}",
        "html": f"""
        <h2>New Lead Handoff</h2>

        <h3>Lead Contact</h3>
        <p><strong>Name:</strong> {contact.get("name") or "Not available"}</p>
        <p><strong>Email:</strong> {contact.get("email") or "Not available"}</p>
        <p><strong>Phone:</strong> {contact.get("phone") or "Not available"}</p>
        <p><strong>Lead ID:</strong> {lead_id}</p>

        <h3>Request</h3>
        <p>{reason}</p>

        <h3>Qualification / Search Preferences</h3>
        <pre>{qualification}</pre>

        {property_html}

        <h3>Latest Message</h3>
        <p>{last_message}</p>
        """,
    }

    return resend.Emails.send(params)