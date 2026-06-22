"""
analytics.py
------------
Local communication scoring and metric compiler.
Zero external API calls — all computed in-process using regex and heuristics.
"""

import re
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TurnLog:
    role: str          # "interviewer" | "candidate"
    text: str
    timestamp: float   # Unix timestamp


@dataclass
class AnalyticsReport:
    total_words: int
    wpm: float
    wpm_label: str
    filler_count: int
    filler_density_pct: float
    filler_breakdown: Dict[str, int]
    technical_accuracy_score: float   # 0-100
    structural_clarity_score: float   # 0-100
    overall_score: float              # 0-100
    overall_label: str                # Excellent / Good / Needs Work
    area_of_improvement: str
    improvement_tips: List[str]
    turns: List[Dict[str, Any]]       # full transcript


# ---------------------------------------------------------------------------
# Filler word patterns
# ---------------------------------------------------------------------------

FILLER_WORDS = ["um", "uh", "like", "basically", "literally", "you know", "sort of", "kind of"]
FILLER_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in FILLER_WORDS) + r")\b",
    re.IGNORECASE
)

# Structural connectors that signal organised thinking
STRUCTURE_CONNECTORS = [
    "firstly", "first", "secondly", "second", "finally", "in conclusion",
    "however", "therefore", "for example", "for instance", "additionally",
    "furthermore", "in summary", "to summarise", "to summarize", "on the other hand",
    "as a result", "because", "since", "although", "specifically",
]
CONNECTOR_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(c) for c in STRUCTURE_CONNECTORS) + r")\b",
    re.IGNORECASE
)

# Technical keywords to check for accuracy overlap with resume
TECH_KEYWORDS = re.compile(
    r"\b(api|model|algorithm|architecture|pipeline|database|framework|library|"
    r"neural|network|training|inference|deployment|docker|kubernetes|git|cloud|"
    r"async|concurrent|rest|graphql|sql|nosql|tensor|layer|epoch|accuracy|"
    r"precision|recall|f1|loss|gradient|backprop|transformer|attention|embedding|"
    r"vector|cluster|class|object|function|module|service|endpoint|cache|index|"
    r"query|dataset|feature|regression|classification|segmentation|detection)\b",
    re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Core scoring functions
# ---------------------------------------------------------------------------

def _candidate_text(turns: List[TurnLog]) -> str:
    """Concatenate all candidate responses into one block."""
    return " ".join(t.text for t in turns if t.role == "candidate")


def _compute_wpm(text: str, elapsed_seconds: float) -> tuple[float, str]:
    word_count = len(text.split())
    elapsed_min = max(elapsed_seconds / 60, 0.01)
    wpm = round(word_count / elapsed_min, 1)

    if wpm < 100:
        label = "Slow"
    elif wpm <= 160:
        label = "Optimal ✅"
    else:
        label = "Fast"

    return wpm, label


def _compute_filler_metrics(text: str) -> tuple[int, float, Dict[str, int]]:
    matches = FILLER_PATTERN.findall(text.lower())
    total_words = max(len(text.split()), 1)
    breakdown = {}
    for m in matches:
        key = m.lower()
        breakdown[key] = breakdown.get(key, 0) + 1

    count = len(matches)
    density_pct = round((count / total_words) * 100, 2)
    return count, density_pct, breakdown


def _compute_technical_accuracy(candidate_text: str, resume_text: str) -> float:
    """
    Heuristic: ratio of tech keywords in answers that also appear in the resume.
    Score 0-100.
    """
    if not resume_text:
        return 50.0

    resume_keywords = set(k.lower() for k in TECH_KEYWORDS.findall(resume_text))
    answer_keywords = set(k.lower() for k in TECH_KEYWORDS.findall(candidate_text))

    if not answer_keywords:
        return 40.0

    overlap = len(answer_keywords & resume_keywords)
    # Bonus for total tech vocabulary used
    vocab_score = min(len(answer_keywords) / 20, 1.0) * 30
    overlap_score = (overlap / max(len(resume_keywords), 1)) * 70

    return round(min(vocab_score + overlap_score, 100), 1)


def _compute_structural_clarity(candidate_text: str) -> float:
    """
    Heuristic based on:
    - Sentence count (enough detail per answer)
    - Average sentence length (not too short/long)
    - Presence of structural connectors
    Score 0-100.
    """
    sentences = re.split(r"[.!?]+", candidate_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    sentence_count = len(sentences)

    if sentence_count == 0:
        return 20.0

    avg_len = sum(len(s.split()) for s in sentences) / sentence_count
    connector_count = len(CONNECTOR_PATTERN.findall(candidate_text))

    # Sentence count score (target 15-40 sentences for a good interview)
    sent_score = min(sentence_count / 30, 1.0) * 40

    # Avg sentence length score (target 10-20 words)
    if 10 <= avg_len <= 20:
        len_score = 30
    elif 7 <= avg_len < 10 or 20 < avg_len <= 25:
        len_score = 20
    else:
        len_score = 10

    # Connector score (target >= 5 connectors)
    conn_score = min(connector_count / 8, 1.0) * 30

    return round(min(sent_score + len_score + conn_score, 100), 1)


def _derive_improvement(tech: float, clarity: float, filler_pct: float) -> tuple[str, List[str]]:
    """Return the weakest area and actionable tips."""
    filler_score = max(0, 100 - filler_pct * 10)  # penalise high filler %

    scores = {
        "Technical Depth": tech,
        "Communication Clarity": clarity,
        "Filler Word Control": filler_score,
    }
    weakest = min(scores, key=scores.get)

    tips_map = {
        "Technical Depth": [
            "Ground your answers with specific technologies, tools, and metrics from your projects.",
            "Use the STAR method (Situation → Task → Action → Result) for technical deep-dives.",
            "Quantify impact: mention latency improvements, accuracy %, scale handled.",
        ],
        "Communication Clarity": [
            "Structure answers using 'First… Then… Finally…' to signal organised thinking.",
            "Aim for 2-4 clear sentences per point before moving on.",
            "Pause briefly before answering — it signals composure and prevents rambling.",
        ],
        "Filler Word Control": [
            "Replace 'um/uh' with a deliberate 1-second pause.",
            "Practice answering mock questions out loud to reduce unconscious fillers.",
            "Record yourself and replay — awareness dramatically reduces filler frequency.",
        ],
    }

    return weakest, tips_map[weakest]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compile_report(
    turns: List[TurnLog],
    elapsed_seconds: float,
    resume_text: str = "",
) -> AnalyticsReport:
    """
    Build the full analytics report from the session turn log.

    Args:
        turns: Ordered list of TurnLog entries.
        elapsed_seconds: Total interview duration in seconds.
        resume_text: Original extracted resume text for keyword overlap scoring.

    Returns:
        A populated AnalyticsReport dataclass.
    """
    logger.info("[Analytics] Compiling interview report...")

    candidate_text = _candidate_text(turns)
    total_words = len(candidate_text.split())

    wpm, wpm_label = _compute_wpm(candidate_text, elapsed_seconds)
    filler_count, filler_pct, filler_breakdown = _compute_filler_metrics(candidate_text)
    tech_score = _compute_technical_accuracy(candidate_text, resume_text)
    clarity_score = _compute_structural_clarity(candidate_text)

    # Overall score = weighted average
    overall = round((tech_score * 0.45) + (clarity_score * 0.35) + (max(0, 100 - filler_pct * 10) * 0.20), 1)
    overall = min(overall, 100)

    if overall >= 75:
        overall_label = "Excellent 🏆"
    elif overall >= 55:
        overall_label = "Good 👍"
    else:
        overall_label = "Needs Work 📈"

    area, tips = _derive_improvement(tech_score, clarity_score, filler_pct)

    report = AnalyticsReport(
        total_words=total_words,
        wpm=wpm,
        wpm_label=wpm_label,
        filler_count=filler_count,
        filler_density_pct=filler_pct,
        filler_breakdown=filler_breakdown,
        technical_accuracy_score=tech_score,
        structural_clarity_score=clarity_score,
        overall_score=overall,
        overall_label=overall_label,
        area_of_improvement=area,
        improvement_tips=tips,
        turns=[asdict(t) for t in turns],
    )

    logger.info(
        f"[Analytics] ✅ Report compiled — Overall: {overall}/100 ({overall_label}), "
        f"WPM: {wpm}, Fillers: {filler_count}"
    )
    return report


def report_to_dict(report: AnalyticsReport) -> Dict[str, Any]:
    """Convert report dataclass to a JSON-serialisable dict."""
    return asdict(report)
