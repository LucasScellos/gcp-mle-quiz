#!/usr/bin/env python3
"""
GCP Professional Machine Learning Engineer - CLI Quiz App
Parses questions from ExamTopic_ML_GCP.pdf and quizzes you interactively.
After each answer, you can ask an LLM to explain why it's correct.
"""

import re
import os
import sys
import json
import random
import textwrap
import pdfplumber
from pathlib import Path

# Load .env file if present
env_path = Path(__file__).parent / ".env"
try:
    from dotenv import load_dotenv
    load_dotenv(env_path)
except ImportError:
    # Manual fallback for .env loading if python-dotenv isn't installed
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip().strip('"').strip("'")

try:
    from google import genai
    from google.genai import types as genai_types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────────────────────
PDF_PATH = Path(__file__).parent / "ExamTopic_ML_GCP.pdf"
CACHE_PATH = Path(__file__).parent / ".questions_cache.json"

# LLM – Google Gemini API via google-genai SDK.
# Gemma 4 is open-weight only (not hosted on the API).
# Override via QUIZ_LLM_MODEL in .env, default = gemini-2.5-flash.
LLM_MODEL = os.getenv("QUIZ_LLM_MODEL", "gemma-4-31b-it")
LLM_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Terminal colours
C = {
    "reset":  "\033[0m",
    "bold":   "\033[1m",
    "dim":    "\033[2m",
    "green":  "\033[92m",
    "red":    "\033[91m",
    "yellow": "\033[93m",
    "cyan":   "\033[96m",
    "blue":   "\033[94m",
    "magenta":"\033[95m",
    "white":  "\033[97m",
}

def c(color: str, text: str) -> str:
    return f"{C[color]}{text}{C['reset']}"


# ── PDF Parsing ───────────────────────────────────────────────────────────────

def extract_questions(pdf_path: Path) -> list[dict]:
    """Extract all MCQ questions from the PDF."""
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                full_text += t + "\n"

    parts = re.split(r"(?=Question \d+:)", full_text)
    questions = []

    for part in parts:
        part = part.strip()
        if not part.startswith("Question"):
            continue

        m = re.match(r"Question (\d+):\s*(.*?)(?=•\s*A\.)", part, re.DOTALL)
        if not m:
            continue

        q_num = int(m.group(1))
        q_text = m.group(2).strip()

        opts = {}
        for letter in ["A", "B", "C", "D"]:
            pattern = rf"•\s*{letter}\.\s*(.*?)(?=•\s*[BCDE]\.|Correct Answer:|$)"
            om = re.search(pattern, part, re.DOTALL)
            if om:
                opts[letter] = re.sub(r"\s+", " ", om.group(1).strip())

        ca = re.search(r"Correct Answer:\s*([A-D])", part)
        if not ca or len(opts) < 2:
            continue

        questions.append({
            "id": q_num,
            "question": re.sub(r"\s+", " ", q_text),
            "options": opts,
            "answer": ca.group(1),
        })

    return questions


def load_questions() -> list[dict]:
    """Load questions from cache or parse from PDF."""
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            return json.load(f)

    print(c("cyan", "📄 Parsing PDF…"), end=" ", flush=True)
    qs = extract_questions(PDF_PATH)
    with open(CACHE_PATH, "w") as f:
        json.dump(qs, f, indent=2)
    print(c("green", f"✓ {len(qs)} questions loaded and cached."))
    return qs


# ── LLM Explanation ───────────────────────────────────────────────────────────

FALLBACK_MODEL = "gemini-2.5-flash"


def _gemini_client():
    """Return a configured google-genai client."""
    if not _GENAI_AVAILABLE:
        raise RuntimeError("google-genai not installed. Run: pip install google-genai")
    return genai.Client(api_key=LLM_API_KEY)


def _list_text_models(client) -> list[str]:
    """List models that support generateContent."""
    try:
        return [
            m.name.replace("models/", "")
            for m in client.models.list()
            if hasattr(m, "supported_actions") and "generateContent" in (m.supported_actions or [])
            or "generate" in (m.name or "")
        ]
    except Exception:
        return []


def _stream_gemini(client, model: str, prompt: str) -> bool:
    """Stream content from Gemini. Returns True on success, False on 404.

    Thinking is disabled to prevent thinking tokens from consuming the
    output budget and causing visible text to be truncated.
    """
    cfg = genai_types.GenerateContentConfig(
        max_output_tokens=4096,
        #thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
    )
    try:
        for chunk in client.models.generate_content_stream(
            model=model, contents=prompt, config=cfg
        ):
            if chunk.text:
                print(chunk.text, end="", flush=True)
        print()
        return True
    except Exception as e:
        msg = str(e)
        if "404" in msg or "NOT_FOUND" in msg:
            return False  # signal to try fallback
        print(c("red", f"\n❌ LLM error: {e}"))
        return True  # non-404 error, don't retry


def _call_with_fallback(prompt: str) -> None:
    """Try LLM_MODEL; on 404, list available models and fall back to FALLBACK_MODEL."""
    client = _gemini_client()
    ok = _stream_gemini(client, LLM_MODEL, prompt)
    if not ok:
        available = _list_text_models(client)
        print(c("yellow", f"\n⚠  Model '{LLM_MODEL}' not found on this API key."))
        if available:
            print(c("dim", "   Available text models:"))
            for m in sorted(available):  # show up to 15
                print(c("dim", f"     • {m}"))
        print(c("cyan", f"   ↩  Retrying with fallback: {FALLBACK_MODEL}\n"))
        _stream_gemini(client, FALLBACK_MODEL, prompt)


def _build_explanation_prompt(question: dict) -> str:
    opts_text = "\n".join(f"  {k}. {v}" for k, v in question["options"].items())
    return (
        f"You are a GCP Professional Machine Learning Engineer exam expert.\n\n"
        f"Question: {question['question']}\n\n"
        f"Options:\n{opts_text}\n\n"
        f"The correct answer is {question['answer']}. "
        f"Explain clearly and concisely why {question['answer']} is correct "
        f"and briefly why the other options are wrong. keep concise in this part "
        f"remember be conciseReference specific GCP services or concepts where relevant. give final exam tip"
    )


def ask_llm(question: dict) -> None:
    """Stream a Gemini explanation for the correct answer."""
    if not LLM_API_KEY:
        print(c("yellow", "⚠  Set GOOGLE_API_KEY in your .env to enable AI explanations."))
        return

    print(c("cyan", "\n💡 Explanation:"))
    print(c("dim", "─" * 70))
    try:
        _call_with_fallback(_build_explanation_prompt(question))
    except Exception as e:
        print(c("red", f"\n❌ LLM error: {e}"))
    print(c("dim", "─" * 70))


def ask_llm_followup(question: dict) -> None:
    """Interactive follow-up question to Gemini."""
    if not LLM_API_KEY:
        print(c("yellow", "⚠  Set GOOGLE_API_KEY in your .env to enable AI explanations."))
        return

    followup = input(c("cyan", "\n❓ Your question: ")).strip()
    if not followup:
        return

    opts_text = "\n".join(f"  {k}. {v}" for k, v in question["options"].items())
    prompt = (
        f"You are a GCP Professional Machine Learning Engineer exam expert. "
        f"Context — the user is studying this question:\n"
        f"Q: {question['question']}\n\nOptions:\n{opts_text}\n"
        f"Correct answer: {question['answer']}.\n\n"
        f"User follow-up: {followup}\n\n"
        f"Answer thoroughly and focus on GCP exam relevance."
    )

    print()
    try:
        _call_with_fallback(prompt)
    except Exception as e:
        print(c("red", f"\n❌ LLM error: {e}"))
    print(c("dim", "─" * 70))


# ── Display Helpers ───────────────────────────────────────────────────────────

def wrap(text: str, width: int = 80, indent: str = "") -> str:
    return textwrap.fill(text, width=width, subsequent_indent=indent)


def print_header(current: int, total: int, score: int, streak: int) -> None:
    pct = int(score / current * 100) if current > 0 else 0
    bar_len = 30
    filled = int(bar_len * current / total)
    bar = "█" * filled + "░" * (bar_len - filled)

    print()
    print(c("bold", "┌─ GCP Pro MLE Quiz " + "─" * 50 + "┐"))
    print(
        f"│  Progress: {c('cyan', bar)} {current}/{total}"
        + f"  Score: {c('green', str(score))}/{current}"
        + f"  ({c('yellow', str(pct) + '%')})"
        + f"  Streak: {c('magenta', '🔥 ' + str(streak) if streak >= 2 else str(streak))}"
    )
    print(c("bold", "└" + "─" * 70 + "┘"))


def print_question(q: dict, num: int, total: int) -> None:
    print(f"\n  {c('dim', 'Q' + str(q['id']))}\n")
    print(f"  {c('bold', wrap(q['question'], width=75, indent='  '))}\n")
    for letter, text in q["options"].items():
        print(f"  {c('cyan', '[' + letter + ']')}  {wrap(text, width=70, indent='       ')}")
    print()


def print_result(correct: bool, answer: str, q: dict) -> None:
    if correct:
        print(c("green", f"\n  ✅  Correct! The answer is {answer}."))
    else:
        print(c("red", f"\n  ❌  Wrong! You chose {answer}. The correct answer is {c('bold', q['answer'])}."))
        print(c("dim", f"     {q['options'][q['answer']]}"))


# ── Session Stats ─────────────────────────────────────────────────────────────

def print_final_stats(score: int, total: int, wrong: list[dict]) -> None:
    pct = int(score / total * 100) if total > 0 else 0
    print()
    print(c("bold", "═" * 70))
    print(c("bold", "  QUIZ COMPLETE"))
    print(c("bold", "═" * 70))
    print(f"  Score:  {c('green', str(score))} / {total}  ({c('yellow', str(pct) + '%')})")

    if pct >= 80:
        print(c("green", "  🎉 Great job! You're on track for the exam."))
    elif pct >= 60:
        print(c("yellow", "  📚 Keep studying – you're getting there!"))
    else:
        print(c("red", "  💪 More practice needed. Review the wrong answers below."))

    if wrong:
        print(c("bold", f"\n  Questions you got wrong ({len(wrong)}):"))
        for q in wrong:
            print(f"    • Q{q['id']}: {q['question'][:70]}…  [Answer: {q['answer']}]")
    print()


# ── Mode Selection ────────────────────────────────────────────────────────────

def choose_mode(questions: list[dict]) -> list[dict]:
    print(c("bold", "\n  🎯 Quiz Mode"))
    print(f"  {c('cyan', '[1]')}  Random – {min(10, len(questions))} random questions")
    print(f"  {c('cyan', '[2]')}  All    – all {len(questions)} questions in order")
    print(f"  {c('cyan', '[3]')}  Range  – pick a question range")
    print(f"  {c('cyan', '[4]')}  Weak   – redo questions from a saved wrong list")
    print()

    while True:
        choice = input(c("white", "  Your choice [1-4]: ")).strip()
        if choice == "1":
            n = input(f"  How many questions? [default 10, max {len(questions)}]: ").strip()
            n = int(n) if n.isdigit() else 20
            n = min(n, len(questions))
            return random.sample(questions, n)
        elif choice == "2":
            return list(questions)
        elif choice == "3":
            ids = [q["id"] for q in questions]
            lo = input(f"  From Q (min {min(ids)}): ").strip()
            hi = input(f"  To   Q (max {max(ids)}): ").strip()
            lo = int(lo) if lo.isdigit() else min(ids)
            hi = int(hi) if hi.isdigit() else max(ids)
            subset = [q for q in questions if lo <= q["id"] <= hi]
            if not subset:
                print(c("red", "  No questions in that range."))
                continue
            return subset
        elif choice == "4":
            wrong_path = Path(__file__).parent / ".wrong_questions.json"
            if not wrong_path.exists():
                print(c("yellow", "  No saved wrong questions yet. Play first!"))
                continue
            with open(wrong_path) as f:
                wrong_ids = set(json.load(f))
            subset = [q for q in questions if q["id"] in wrong_ids]
            print(c("cyan", f"  Loaded {len(subset)} wrong questions."))
            return subset
        else:
            print(c("red", "  Please enter 1, 2, 3, or 4."))


# ── Main Quiz Loop ────────────────────────────────────────────────────────────

def run_quiz(questions: list[dict]) -> None:
    score = 0
    streak = 0
    wrong_questions = []
    total = len(questions)

    for i, q in enumerate(questions):
        print_header(i, total, score, streak)
        print_question(q, i + 1, total)

        # Get answer
        while True:
            raw = input(c("white", "  Your answer [A/B/C/D] or [S]kip / [Q]uit: ")).strip().upper()
            if raw in ("A", "B", "C", "D", "S", "Q"):
                break
            print(c("red", "  Please enter A, B, C, D, S, or Q."))

        if raw == "Q":
            print(c("yellow", "\n  👋 Quitting early…"))
            break
        elif raw == "S":
            print(c("dim", "  ⏭  Skipped."))
            streak = 0
            continue

        # Evaluate
        correct = raw == q["answer"]
        print_result(correct, raw, q)

        if correct:
            score += 1
            streak += 1
            if streak >= 3:
                print(c("magenta", f"  🔥 {streak}-question streak!"))
        else:
            streak = 0
            wrong_questions.append(q)

        # Post-answer options
        print(
            f"\n  {c('dim', '[E]')} Explain with AI  "
            f"{c('dim', '[F]')} Follow-up question  "
            f"{c('dim', '[N]')} Next"
        )
        while True:
            action = input("  > ").strip().upper()
            if action in ("E", ""):
                ask_llm(q)
                print(
                    f"\n  {c('dim', '[F]')} Follow-up  {c('dim', '[N]')} Next"
                )
            elif action == "F":
                ask_llm_followup(q)
            elif action == "N":
                break
            else:
                break

    # Save wrong questions for "Weak" mode – merge with existing, don't overwrite
    wrong_path = Path(__file__).parent / ".wrong_questions.json"
    existing_ids: set = set()
    if wrong_path.exists():
        with open(wrong_path) as f:
            existing_ids = set(json.load(f))
    new_ids = {q["id"] for q in wrong_questions}
    merged_ids = sorted(existing_ids | new_ids)
    with open(wrong_path, "w") as f:
        json.dump(merged_ids, f)

    print_final_stats(score, i + 1, wrong_questions)


# ── Entry Point ───────────────────────────────────────────────────────────────

def main() -> None:
    # Banner
    print()
    print(c("blue", "╔══════════════════════════════════════════════════════════════════════╗"))
    print(c("blue", "║") + c("bold", "        GCP Professional Machine Learning Engineer — Quiz CLI        ") + c("blue", "║"))
    print(c("blue", "╚══════════════════════════════════════════════════════════════════════╝"))
    print()

    if not PDF_PATH.exists():
        print(c("red", f"❌ PDF not found: {PDF_PATH}"))
        sys.exit(1)

    if not _GENAI_AVAILABLE:
        print(c("yellow", "⚠  google-genai not installed. Run: pip install google-genai"))
    elif not LLM_API_KEY:
        print(c("yellow", "⚠  GOOGLE_API_KEY not set – AI explanations will be disabled."))
        print(c("dim", "   Add it to your .env file: GOOGLE_API_KEY=AIza…\n"))
    else:
        print(c("green", f"✓ Gemini ready ({LLM_MODEL})"))

    questions = load_questions()
    print(c("green", f"✓ {len(questions)} questions available\n"))

    while True:
        subset = choose_mode(questions)
        run_quiz(subset)

        again = input(c("white", "\n  Play again? [Y/n]: ")).strip().lower()
        if again == "n":
            break

    print(c("cyan", "\n  Good luck on the exam! 🚀\n"))


if __name__ == "__main__":
    main()
