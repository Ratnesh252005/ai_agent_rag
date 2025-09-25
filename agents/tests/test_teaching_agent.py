import random

def teach(structured: dict, mode: str = "explain", options: dict = None) -> dict:
    """
    Teaching agent that can explain, summarize, or create quizzes 
    from structured knowledge input.
    """
    options = options or {}
    topic = structured.get("topic", "Unknown topic")
    definition = structured.get("definition", "")
    steps = structured.get("steps", [])
    examples = structured.get("examples", [])
    summary = structured.get("summary", "")

    if mode == "explain":
        content = f"Let's learn about {topic}.\n\n{definition}\n"
        if steps:
            content += "Key steps:\n" + "\n".join(f"- {s}" for s in steps)
        if examples:
            content += f"\nExample: {examples[0]}"
        return {"mode": "explain", "content": content}

    elif mode == "summary":
        max_lines = options.get("summary_lines", 2)
        content = f"Summary of {topic}: {summary or definition}"
        lines = content.split(".")
        trimmed = ". ".join(lines[:max_lines])
        return {"mode": "summary", "content": trimmed}

    elif mode == "quiz":
        num_qs = options.get("num_questions", 2)
        quiz = []
        if definition:
            quiz.append(f"What is {topic}?")
        if steps:
            quiz.append(f"List one step involved in {topic}.")
        if examples:
            quiz.append(f"Give an example of {topic}.")
        random.shuffle(quiz)
        return {"mode": "quiz", "quiz": quiz[:num_qs]}

    else:
        return {"mode": "unknown", "content": "I cannot handle this mode yet."}
