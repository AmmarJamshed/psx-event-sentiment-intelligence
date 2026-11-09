from __future__ import annotations

import re
from dataclasses import dataclass


POSITIVE = {
    "optimism": ["optimistic", "hope", "hopeful", "bullish", "rally", "surge", "rebound", "confidence", "stable", "reform success", "breakthrough"],
    "confidence": ["confidence", "reassure", "stability", "strong", "support", "inflows", "upgrade", "approval", "agreement reached"],
    "investor_enthusiasm": ["enthusiasm", "buying interest", "record high", "all-time high", "blue chip rally", "foreign buying"],
}

NEGATIVE = {
    "fear": ["fear", "scared", "alarm", "threat", "risk", "danger", "crisis"],
    "panic": ["panic", "crash", "collapse", "rout", "selloff", "sell-off", "plunge"],
    "uncertainty": ["uncertainty", "unclear", "speculation", "rumours", "rumors", "volatile", "instability"],
    "anger": ["anger", "outrage", "protest", "clash", "condemn"],
    "political_dissatisfaction": ["no-confidence", "dissolution", "impeachment", "rigging", "deadlock", "martial", "arrest of"],
    "economic_anxiety": ["inflation", "devaluation", "default", "imf pressure", "reserves fall", "circular debt", "fuel shortage", "austerity"],
    "investor_pessimism": ["bearish", "outflows", "downgrade", "loss", "decline", "weak", "slump"],
}

# Words that often describe market moves themselves — down-weight as *public* sentiment evidence
MARKET_DESCRIPTORS = {
    "index fell",
    "index rose",
    "kse-100",
    "points",
    "volume clocked",
    "market closed",
    "bears dominated",
    "bulls dominated",
}


@dataclass
class SentimentResult:
    score: float
    pos_intensity: float
    neg_intensity: float
    emotions: dict[str, float]
    news_volume: int
    disagreement: float
    notes: str


def _count_hits(text: str, phrases: list[str]) -> int:
    t = text.lower()
    return sum(1 for p in phrases if p in t)


def score_texts(texts: list[str]) -> SentimentResult:
    """Lexicon multi-emotion scorer for English / Roman Urdu keyword overlap.

    Returns null-equivalent NaNs via empty input rather than inventing history.
    """
    if not texts:
        return SentimentResult(
            score=float("nan"),
            pos_intensity=float("nan"),
            neg_intensity=float("nan"),
            emotions={k: float("nan") for k in list(POSITIVE) + list(NEGATIVE)},
            news_volume=0,
            disagreement=float("nan"),
            notes="No texts available; sentiment left missing (not fabricated).",
        )

    emotions = {k: 0.0 for k in list(POSITIVE) + list(NEGATIVE)}
    scores = []
    for text in texts:
        clean = re.sub(r"\s+", " ", str(text)).strip()
        if not clean:
            continue
        # Penalize pure market-move regurgitation
        marketish = sum(1 for m in MARKET_DESCRIPTORS if m in clean.lower())
        pos = 0
        neg = 0
        for emo, phrases in POSITIVE.items():
            hits = _count_hits(clean, phrases)
            emotions[emo] += hits
            pos += hits
        for emo, phrases in NEGATIVE.items():
            hits = _count_hits(clean, phrases)
            emotions[emo] += hits
            neg += hits
        raw = (pos - neg) / max(pos + neg, 1)
        if marketish:
            raw *= 0.5
        scores.append(max(-1.0, min(1.0, raw)))

    if not scores:
        return score_texts([])

    import numpy as np

    arr = np.array(scores, dtype=float)
    total_hits = sum(emotions.values()) or 1.0
    emo_norm = {k: float(v / total_hits) for k, v in emotions.items()}
    pos_intensity = float(np.mean(np.clip(arr, 0, 1)))
    neg_intensity = float(np.mean(np.clip(-arr, 0, 1)))
    return SentimentResult(
        score=float(arr.mean()),
        pos_intensity=pos_intensity,
        neg_intensity=neg_intensity,
        emotions=emo_norm,
        news_volume=len(scores),
        disagreement=float(arr.std()) if len(arr) > 1 else 0.0,
        notes="Lexicon-based contemporaneous scoring; not a historical reconstruction.",
    )
