# GCP Pro MLE — CLI Quiz App

Interactive CLI quiz to prepare for the **GCP Professional Machine Learning Engineer** exam.

## Setup

```bash
# Create and activate virtualenv (already done)
python3 -m venv venv && source venv/bin/activate

# Install dependencies
pip install pdfplumber google-genai python-dotenv
```

## Configuration

Add your Google API key to `.env` (already set up):

```env
GOOGLE_API_KEY=AIza...
```

Get a free key at → https://aistudio.google.com/apikey

## Run

```bash
source venv/bin/activate
python3 quiz.py
```

## Features

| Feature | Description |
|---|---|
| 📄 **277 questions** | Auto-parsed from examtopics pdf, cached after first run |
| 🎯 **4 modes** | Random N, All in order, Range (Q200–Q285…), Weak spots |
| ✅ **Instant feedback** | Correct/incorrect with the right answer shown |
| 💡 **AI explanation** | Press `E` after any answer — Gemini explains why the answer is correct |
| ❓ **Follow-up** | Press `F` to ask Gemini a custom follow-up question |
| 🔥 **Streak counter** | Track consecutive correct answers |
| 💾 **Weak mode** | Wrong answers saved to `.wrong_questions.json`, replayable |

## Controls

| Key | Action |
|---|---|
| `A` / `B` / `C` / `D` | Submit your answer |
| `S` | Skip question |
| `Q` | Quit quiz |
| `E` (after answer) | Get Gemini explanation |
| `F` (after answer) | Ask a follow-up question |
| `N` (after answer) | Go to next question |

