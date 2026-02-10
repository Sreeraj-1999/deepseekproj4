# # llm_service.py
# import requests
# from typing import List, Dict
# import re

# def generate_email_reply(email_context: Dict, context_docs: List[Dict], qwen_url: str) -> str:
#     """Generate email reply - let LLM figure out the approach"""
    
#     # Extract email details
#     email_body = email_context.get('body', '')
#     sender_name = email_context.get('sender_name', '')
#     sender_email = email_context.get('sender_email', '')
#     sender_company = email_context.get('sender_company', '')
#     subject = email_context.get('subject', '')
    
#     # Determine greeting
#     greeting = determine_greeting(sender_name, sender_company)
    
#     # Build technical context if available and relevant
#     technical_context = ""
#     if context_docs and context_docs[0].get('score', 0) > 0.5:
#         technical_context = "\n\nRELEVANT EQUIPMENT KNOWLEDGE (use if applicable):\n"
#         for doc in context_docs[:3]:
#             technical_context += f"\n• Equipment: {doc['equipment']}"
#             technical_context += f"\n  Issue: {doc['issue']}"
#             technical_context += f"\n  Solution: {doc['solution']}"
#             if doc.get('suspected_parts'):
#                 technical_context += f"\n  Parts: {doc['suspected_parts']}"
#             technical_context += "\n"
    
#     # Simple, trust-the-LLM prompt
#     system_prompt = """You are a senior marine operations and engineering support specialist who writes professional email replies for clients, vessels, owners, and fleet management teams.

# Write a complete, professional email response. 

# Structure:
# - Appropriate greeting (will be provided)
# - Acknowledge their concern/request
# - Never admit internal fault, failure, or negligence.
# - Provide helpful guidance or next steps
# - Professional sign-off

# Guidelines:
# - Identify the email's nature (complaint, escalation, technical issue, request, operational concern).
# - If complaint/escalation:
#     • Show empathy without admitting fault.
#     • Do NOT suggest actions the client must take.
#     • Do NOT blame internal teams or SOPs.
#     • Do NOT make commitments like “we will escalate immediately” unless explicitly asked.
#     • Provide calm, structured reassurance and a safe action plan.
#     • Never invent timelines, deadlines, guarantees, or escalation promises.
#     • Do NOT imply internal mistakes indirectly (e.g., “this should have been checked”, “as per SOPs”, “we should have verified this earlier”).
#     • Match the seriousness of the situation when acknowledging their concern.
#     • Do NOT give instructions to the vessel or crew (e.g., “ensure crew does…”, “train staff”, “perform inspections”).
#     • Do NOT propose new procedures, trainings, checklists, or operational changes unless explicitly requested.
#     • Do NOT provide phone numbers, emails, or contact details unless included in the original email.
#     • Never introduce invented steps, bullet lists, action plans, or operational workflows unless they were specifically mentioned by the sender.
#     • Respond using a concise narrative paragraph style, not bullet points, unless the original email uses bullet points.


# - If technical:
#     • Give logical next steps without inventing data or timelines.
# - If operational concern:
#     • Acknowledge impact and provide realistic next steps.
# - Reference technical context ONLY if relevant.
# - Keep the email concise, confident, and professional.
# - End with: “Best regards,\nMarine Engineering Support Team”.


# Do NOT add "Subject:" lines. Do NOT repeat greeting/sign-off."""

#     user_prompt = f"""Email from: {sender_name if sender_name else sender_email}
# {f'Company: {sender_company}' if sender_company else ''}
# Subject: {subject if subject else '(no subject)'}

# Email body:
# {email_body}
# {technical_context}

# Greeting to use: {greeting}

# Write the complete email reply:"""

#     try:
#         response = requests.post(
#             qwen_url,
#             json={
#                 "messages": [
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_prompt}
#                 ],
#                 "response_type": "email_reply"
#             },
#             timeout=60
#         )
        
#         response.raise_for_status()
#         result = response.json()
#         draft = result.get('response', '').strip()
        
#         # Minimal cleanup
#         draft = clean_email_draft(draft, greeting)
        
#         return draft
        
#     except Exception as e:
#         # Simple fallback
#         return f"{greeting}\n\nThank you for your email. We have received your message and will respond with detailed information shortly.\n\nBest regards,\nMarine Engineering Support Team"


# def determine_greeting(sender_name: str, sender_company: str) -> str:
#     """Determine appropriate greeting"""
    
#     if sender_name:
#         name_parts = sender_name.strip().split()
#         if len(name_parts) > 0:
#             first_name = name_parts[0]
#             # Check for titles
#             if first_name.rstrip('.').lower() in ['mr', 'mrs', 'ms', 'dr', 'capt', 'captain', 'chief', 'eng', 'engr']:
#                 if len(name_parts) > 1:
#                     return f"Dear {' '.join(name_parts)},"
#                 return "Dear Sir/Madam,"
#             return f"Dear {first_name},"
    
#     if sender_company:
#         return f"Dear {sender_company} Team,"
    
#     return "Dear Sir/Madam,"


# def clean_email_draft(draft: str, expected_greeting: str) -> str:
#     """Minimal cleanup - just remove obvious duplicates"""
    
#     # Remove "Subject:" lines
#     draft = re.sub(r'^Subject:.*?\n\n?', '', draft, flags=re.IGNORECASE | re.MULTILINE)
    
#     # Remove duplicate consecutive greetings
#     lines = draft.split('\n')
#     cleaned = []
#     prev_was_greeting = False
    
#     for line in lines:
#         is_greeting = line.lower().strip().startswith(('dear ', 'hello ', 'hi '))
#         if is_greeting and prev_was_greeting:
#             continue
#         cleaned.append(line)
#         prev_was_greeting = is_greeting
    
#     draft = '\n'.join(cleaned)
    
#     # Ensure greeting at start if completely missing
#     first_line = draft.split('\n')[0].lower().strip()
#     if not first_line.startswith(('dear', 'hello', 'hi')):
#         draft = f"{expected_greeting}\n\n{draft}"
    
#     # Ensure sign-off exists
#     if not re.search(r'(best regards|sincerely|regards)', draft, re.IGNORECASE):
#         draft += "\n\nBest regards,\nMarine Engineering Support Team"
    
#     # Clean excessive newlines
#     draft = re.sub(r'\n{3,}', '\n\n', draft)
    
#     return draft.strip()






# llm_service.py
import requests
from typing import List, Dict
import re

def generate_email_reply(email_context: Dict, context_docs: List[Dict], qwen_url: str) -> str:
    """Generate email reply - LLM decides tone + structure."""

    # Extract email details
    email_body = email_context.get('body', '')
    sender_name = email_context.get('sender_name', '')
    sender_email = email_context.get('sender_email', '')
    sender_company = email_context.get('sender_company', '')
    subject = email_context.get('subject', '')

    # Greeting
    greeting = determine_greeting(sender_name, sender_company)

    # Determine first name for sign-off
    signoff_name = sender_name.split()[0] if sender_name else ""

    # Optional technical context
    technical_context = ""
    if context_docs and context_docs[0].get('score', 0) > 0.5:
        technical_context = "\n\nRELEVANT CONTEXT (use only if actually helpful):\n"
        for doc in context_docs[:3]:
            technical_context += f"\n• Equipment: {doc['equipment']}"
            technical_context += f"\n  Issue: {doc['issue']}"
            technical_context += f"\n  Solution: {doc['solution']}"
            if doc.get('suspected_parts'):
                technical_context += f"\n  Parts: {doc['suspected_parts']}"
            technical_context += "\n"

    # STRICT short prompt for Qwen 7B (no hallucinations)
    system_prompt = """
Write a short, professional email reply.

STRICT RULES (DO NOT BREAK):
- Do NOT invent: facts, timelines, actions, inspections, procedures, names, phone numbers, emails, websites, departments, or regulations.
- Do NOT mention training, checklists, maintenance schedules, or communication protocols unless the sender explicitly mentioned them.
- Do NOT assign tasks to vessel/crew unless the sender clearly asked.
- Do NOT use bullet points unless the sender used bullet points.
- Do NOT admit fault or imply internal mistakes.
- Do NOT fabricate events or technical steps.
- Do NOT add P.S., extra signatures, contact details, or anything beyond the required sign-off.

WHAT YOU MUST DO:
- Acknowledge the concern calmly and respectfully.
- Keep the reply in 1–2 short paragraphs.
- Provide only general reassurance or a neutral next step.
- Use the greeting provided.
- End with EXACTLY:

Best regards,
<NAME>
Marine Engineering Support Team
"""

    # User prompt
    user_prompt = f"""
Email from: {sender_name or sender_email}
Company: {sender_company or ''}
Subject: {subject or '(no subject)'}

Email body:
{email_body}

{technical_context}

Greeting to use: {greeting}
Signoff name to use: {signoff_name}

Write the complete email reply:
"""

    try:
        response = requests.post(
            qwen_url,
            json={
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            },
            timeout=60
        )

        response.raise_for_status()
        result = response.json()
        draft = result.get('response', '').strip()

        draft = clean_email_draft(draft, greeting, signoff_name)
        return draft

    except Exception:
        if signoff_name:
            return f"{greeting}\n\nThank you for your email. We will revert shortly.\n\nBest regards,\n{signoff_name}\nMarine Engineering Support Team"
        return f"{greeting}\n\nThank you for your email. We will revert shortly.\n\nBest regards,\nMarine Engineering Support Team"


def determine_greeting(sender_name: str, sender_company: str) -> str:
    """Determine appropriate greeting"""
    if sender_name:
        parts = sender_name.strip().split()
        first = parts[0]
        titles = ['mr', 'mrs', 'ms', 'dr', 'capt', 'captain', 'chief', 'eng', 'engr']
        if first.rstrip('.').lower() in titles:
            return f"Dear {sender_name},"
        return f"Dear {first},"
    if sender_company:
        return f"Dear {sender_company} Team,"
    return "Dear Sir/Madam,"


def clean_email_draft(draft: str, expected_greeting: str, signoff_name: str) -> str:
    """Cleanup for email formatting"""

    # Remove Subject:
    draft = re.sub(r'^Subject:.*?\n\n?', '', draft, flags=re.IGNORECASE | re.MULTILINE)

    # Remove duplicate greetings
    lines = draft.split('\n')
    cleaned = []
    seen_greeting = False

    for line in lines:
        lower = line.lower().strip()
        is_greeting = lower.startswith(('dear ', 'hello ', 'hi '))
        if is_greeting:
            if seen_greeting:
                continue
            seen_greeting = True
        cleaned.append(line)

    draft = '\n'.join(cleaned).strip()

    # Ensure greeting exists
    first_line = draft.split('\n')[0].lower()
    if not first_line.startswith(('dear ', 'hello', 'hi')):
        draft = f"{expected_greeting}\n\n{draft}"

    # Enforce correct sign-off (remove any model-generated ones)
    draft = re.sub(
        r"best regards[\s\S]*",
        "",
        draft,
        flags=re.IGNORECASE
    ).strip()

    # Apply correct final sign-off
    if signoff_name:
        draft += f"\n\nBest regards,\n{signoff_name}\nMarine Engineering Support Team"
    else:
        draft += "\n\nBest regards,\nMarine Engineering Support Team"

    # Reduce excessive blank lines
    draft = re.sub(r'\n{3,}', '\n\n', draft)

    return draft.strip()

