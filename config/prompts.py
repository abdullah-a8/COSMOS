RAG_SYSTEM_PROMPT = """
Answer the question based on the context below. 

Important instructions:
1. If you can't answer the question based on the provided context, reply "I need more context".
2. Whenever you use information from the context, you MUST cite the original source properly.
3. CITATION FORMAT INSTRUCTIONS:
   - For websites: Use only the domain name as the citation text (e.g., [Source: Wccftech])
   - For PDFs: Use "PDF document" followed by the ID (e.g., [Source: PDF document e1f2...])
   - For YouTube: Use "YouTube video" followed by the ID in parentheses (e.g., [Source: YouTube video (abc123)])
   - DO NOT include long URLs or section numbers in the visible citation
4. Each extract has a "SOURCE FOR EXTRACT #N" line following it - use that information to create your citation.
5. If multiple sources support your answer, cite all relevant sources.
6. Don't make up information that isn't in the context.

Context:
{context}

Question: {question}
"""

# Fast RAG prompt - more concise for quicker generation
RAG_FAST_PROMPT = """Answer the question concisely using only this context:
{context}

Question: {question}
Answer:"""

# --- Gmail Agent Prompts ---

GMAIL_CLASSIFY_PROMPT = """
Please classify the following email into one of these categories:
- Support (technical help, troubleshooting)
- Sales (inquiries about purchasing, pricing questions)
- Personal (non-business communication)
- Information (general information requests)
- Urgent (time-sensitive matters)
- Other (anything that doesn't fit above)

Subject: {email_subject}

Email Body:
{email_body}

Return only the category name without any explanation.
"""

GMAIL_GENERATE_REPLY_PROMPT = """
Generate a reply to the following email:

From: {sender_name}
Subject: {subject}

Email Body:
{email_body}

--- 
Reply Parameters:
- Tone: {tone} (e.g., formal, friendly, direct)
- Style: {style} (e.g., concise, detailed, professional)
- Length: {length} (e.g., brief, standard, comprehensive)

--- 
Optional User Context (Use this information to shape the reply if provided):
{user_context}
---

Write a complete, ready-to-send email response. Don't use placeholders and make it sound natural based on the original email and any user context provided.
"""

GMAIL_SUMMARIZE_PROMPT = """
Please provide a concise, personalized summary of the key points from the following email content.
Focus on the main topic, decisions made, and any action items mentioned.

Recipient Info:
Name: {recipient_name}
Email: {recipient_email}

Email Content:
{email_content}

Instructions:
1. Use the recipient's name when referring to them in the summary
2. Identify if the email is directly addressed to the recipient or if they are CC'd/BCC'd
3. Highlight any specific actions or responses required from the recipient
4. Note any deadlines or time-sensitive information relevant to the recipient
5. Replace generic references with personalized ones (e.g., "you" instead of "the recipient")

Summary:
"""