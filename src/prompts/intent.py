"""
Prompt template for classifying user intent in the interactive feedback loop.

Used by: src/agent/nodes.py (human_feedback_loop node, Phase 5)
"""

INTENT_CLASSIFICATION_SYSTEM_PROMPT = """\
You are an intent classifier for a hiring assistant chatbot. Given a user's \
message, classify it into exactly one of the following intent categories:

## Intent Categories

- **refine**: User wants to modify, add, or remove job requirements. \
  Examples: "Drop AWS", "Add TypeScript", "Make React less important", \
  "Change experience to 3 years", "Remove the PhD requirement"
- **compare**: User wants to compare two or more candidates. \
  Examples: "Compare the top 3", "Compare Alice and Bob", \
  "How do the top 2 stack up?", "Side by side comparison"
- **questions**: User wants interview questions for a candidate. \
  Examples: "Generate questions for Alice", "What should I ask candidate 1?", \
  "Interview questions for the top candidate"
- **explain**: User wants to understand WHY a candidate was ranked a certain way. \
  Examples: "Why did Alice rank higher?", "Explain the ranking", \
  "Why is Bob ranked lower than Alice?", "What makes the top candidate the best?"
- **report**: User wants to see or regenerate match reports. \
  Examples: "Show me the report", "Generate reports", "Full report for Alice"
- **done**: User wants to end the conversation. \
  Examples: "done", "that's all", "thank you", "I'm finished", "exit"
- **new_search**: User wants to start over with a new job description. \
  Examples: "New search", "Let me paste a new JD", "Start over", "Reset"

## Rules

1. If the message is ambiguous, choose the most likely intent.
2. If none of the above fit, default to "explain" (user probably wants more info).
3. Respond with ONLY the intent keyword, nothing else.
"""

# Keyword-based fallback patterns for when LLM is unavailable.
# Order matters — first match wins.
KEYWORD_INTENT_MAP: list[tuple[list[str], str]] = [
    (["compare", "versus", "vs", "side by side", "stack up", "head to head"], "compare"),
    (["question", "interview", "ask", "what should i ask"], "questions"),
    (["why", "explain", "reason", "ranking", "ranked higher", "ranked lower", "what makes"], "explain"),
    (["drop", "add", "remove", "change", "modify", "update", "refine", "less important", "more important", "lower", "raise", "increase", "decrease"], "refine"),
    (["report", "show me", "full report", "match report"], "report"),
    (["new search", "start over", "reset", "new jd", "paste a new", "different job"], "new_search"),
    (["done", "that's all", "thank", "finished", "exit", "quit", "goodbye", "bye"], "done"),
]


def build_intent_classification_prompt(human_feedback: str) -> list[dict[str, str]]:
    """Build the message list for intent classification.

    Args:
        human_feedback: The user's latest message.

    Returns:
        List of message dicts for LLM invocation.
    """
    return [
        {"role": "system", "content": INTENT_CLASSIFICATION_SYSTEM_PROMPT},
        {"role": "user", "content": f"Classify this user message:\n\n\"\"\"\n{human_feedback}\n\"\"\""},
    ]
