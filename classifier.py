"""
Heuristic symptom pattern classifier (non-diagnostic, EN only)
-------------------------------------------------------------
This module produces probabilistic pattern estimates for broad, non-diagnostic
symptom clusters. It does NOT output medical diagnoses. All probabilities are
derived from transparent keyword weighting with basic negation handling.

Current features:
1. English-only keyword dictionaries (weighted) per cluster.
2. Text aggregation from the structured record built in chat.py.
3. Normalization + accent stripping + lowercase.
4. Negation detection ("no", "not", "denies", "without") within a narrow token window.
5. Weighted sum -> softmax probabilities.
6. Probability filtering: hide very low-probability clusters (< MIN_PROB_TO_SHOW) except top-1.

Planned extension points:
- Replace heuristics with trainable embedding or transformer classifier.
- Calibrated probabilities (e.g., temperature scaling / isotonic).
- Explanations via SHAP/LIME when using ML model.
- Temporal/severity/intensity weighting.
"""
from __future__ import annotations
import math
import re
import unicodedata
from typing import Dict, List, Any, Tuple

# CONFIGURATION CONSTANTS
MIN_PROB_TO_SHOW = 0.04          # Hide clusters below this probability (except top-1)
NEGATION_WINDOW_TOKENS = 4       # Max tokens between negation cue and target keyword
SOFTMAX_TEMPERATURE = 1.0        # Adjust >1 to flatten, <1 to sharpen

# Weighted English keyword dictionaries per cluster (already lowercase, no accents needed)
# Weights: 2.0 = strong core, 1.5 = medium, 1.0 = supporting.
CLUSTERS: Dict[str, Dict[str, Any]] = {
    "Respiratory_Syndrome": {
        "keywords": {
            "cough": 2.0,
            "shortness of breath": 2.0,
            "wheeze": 1.5,
            "chest tightness": 1.5,
            "sputum": 1.0,
            "dyspnea": 2.0,
            "phlegm": 1.0,
        },
        "description": "Respiratory pattern (cough, breathing difficulty, wheeze)."
    },
    "Gastrointestinal_Syndrome": {
        "keywords": {
            "abdominal pain": 2.0,
            "stomach pain": 2.0,
            "nausea": 1.5,
            "vomiting": 2.0,
            "diarrhea": 2.0,
            "constipation": 1.5,
            "bloating": 1.0,
            "loss of appetite": 1.5,
        },
        "description": "Gastrointestinal pattern (abdominal discomfort, GI upset)."
    },
    "Neurological_Syndrome": {
        "keywords": {
            "headache": 2.0,
            "dizziness": 1.5,
            "weakness": 2.0,
            "numbness": 2.0,
            "tingling": 1.5,
            "blurred vision": 1.5,
            "seizure": 2.0,
            "memory loss": 1.0,
        },
        "description": "Neurological pattern (headache, focal or sensory changes)."
    },
    "Systemic_Fever_Inflammatory": {
        "keywords": {
            "fever": 2.0,
            "chills": 1.5,
            "sweating": 1.0,
            "night sweats": 1.5,
            "malaise": 1.0,
            "fatigue": 1.5,
        },
        "description": "Systemic inflammatory pattern (feverish or constitutional)."
    },
    "Musculoskeletal_Pain": {
        "keywords": {
            "joint pain": 2.0,
            "muscle pain": 2.0,
            "stiffness": 1.5,
            "back pain": 2.0,
            "spasm": 1.0,
            "tenderness": 1.0,
        },
        "description": "Musculoskeletal pattern (joint / muscle symptoms)."
    },
    "Cardiovascular_Concern": {
        "keywords": {
            "chest pain": 2.0,
            "palpitations": 2.0,
            "syncope": 2.0,
            "fainting": 2.0,
            "exertional dyspnea": 2.0,
            "leg swelling": 1.5,
        },
        "description": "Cardiovascular pattern (chest pain, palpitations, syncope)."
    },
    "Dermatologic_Irritation": {
        "keywords": {
            "rash": 2.0,
            "itching": 1.5,
            "redness": 1.5,
            "skin lesion": 2.0,
            "hives": 2.0,
            "scaling": 1.0,
        },
        "description": "Dermatologic pattern (rash, pruritus, lesions)."
    },
    "Anxiety_Stress_Presentation": {
        "keywords": {
            "anxiety": 2.0,
            "nervousness": 1.5,
            "insomnia": 2.0,
            "panic": 2.0,
            "hyperventilation": 1.5,
            "stress": 1.5,
        },
        "description": "Anxiety / stress related pattern (affective / autonomic)."
    },
    "Metabolic_General": {
        "keywords": {
            "thirst": 1.5,
            "excessive thirst": 2.0,
            "weight loss": 2.0,
            "fatigue": 1.5,
            "polyuria": 2.0,
            "increased urination": 2.0,
            "appetite loss": 1.5,
        },
        "description": "Metabolic / general pattern (energy, weight, fluid changes)."
    },
    "Other_Unclassified": {
        "keywords": {},
        "description": "Fallback when no clear pattern signals present."
    }
}

DISCLAIMER = "Probabilistic pattern estimation ONLY. This is NOT a medical diagnosis. Seek professional evaluation."


def _normalize(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')
    return text


def _collect_text(record: Dict[str, Any]) -> str:
    """Aggregate relevant free-text fields from the structured record.

    NOTE: Field names remain in Spanish (legacy schema) but content is expected
    to be predominantly English going forward. Future refactor may rename keys.
    """
    parts: List[str] = []
    mc = record.get("motivo_consulta")
    if mc:
        parts.append(str(mc))
    ea = record.get("enfermedad_actual", {})
    if isinstance(ea, dict):
        for k in ("sintoma_principal", "inicio"):
            v = ea.get(k)
            if isinstance(v, dict):
                parts.append(" ".join(str(x) for x in v.values()))
            elif v:
                parts.append(str(v))
        car = ea.get("caracteristicas", [])
        if isinstance(car, list):
            parts.extend([str(c) for c in car])
    for k in ("sintomas_asociados", "antecedentes_personales", "antecedentes_familiares"):
        arr = record.get(k, [])
        if isinstance(arr, list):
            parts.extend([str(x) for x in arr])
    return _normalize(" | ".join(parts))


NEGATION_CUES = {"no", "not", "denies", "without"}

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())

def _is_negated(normalized_text: str, keyword: str) -> bool:
    """Return True if keyword appears in a negated context.

    Simple sliding window: if a negation cue occurs within NEGATION_WINDOW_TOKENS
    tokens before the first token of the keyword span, we mark it negated.
    """
    tokens = _tokenize(normalized_text)
    kw_tokens = _tokenize(keyword)
    if not kw_tokens:
        return False
    # Build index of positions where keyword sequence matches
    for i in range(0, len(tokens) - len(kw_tokens) + 1):
        if tokens[i:i+len(kw_tokens)] == kw_tokens:
            start = i
            window_start = max(0, start - NEGATION_WINDOW_TOKENS)
            context_slice = tokens[window_start:start]
            if any(t in NEGATION_CUES for t in context_slice):
                return True
    return False

def _apply_softmax(scores: List[float], temperature: float = 1.0) -> List[float]:
    if temperature <= 0:
        temperature = 1.0
    scaled = [s / temperature for s in scores]
    max_s = max(scaled) if scaled else 0.0
    exps = [math.exp(s - max_s) for s in scaled]
    denom = sum(exps) or 1.0
    return [e / denom for e in exps]


def classify_case(structured_record: Dict[str, Any]) -> Dict[str, Any]:
    norm_text = _collect_text(structured_record)  # already normalized
    cluster_results = []
    any_positive = False
    for name, spec in CLUSTERS.items():
        kw_map: Dict[str, float] = spec.get("keywords", {})
        matched: List[Tuple[str, float]] = []
        score = 0.0
        for kw, weight in kw_map.items():
            nkw = _normalize(kw)
            if nkw and nkw in norm_text:
                if _is_negated(norm_text, kw):
                    continue  # skip negated mention
                score += weight
                matched.append((kw, weight))
        if score > 0 and name != "Other_Unclassified":
            any_positive = True
        cluster_results.append({
            "name": name,
            "score": round(score, 3),
            "matched_terms": [f"{k} (w={w})" for k, w in matched],
            "description": spec["description"]
        })
    # If no positives at all, give minimal score to Other_Unclassified
    if not any_positive:
        for c in cluster_results:
            if c["name"] == "Other_Unclassified":
                c["score"] = 1.0
                break
    scores = [c["score"] for c in cluster_results]
    if all(s == 0 for s in scores):
        probs = [1.0 / len(scores)] * len(scores)
    else:
        probs = _apply_softmax(scores, SOFTMAX_TEMPERATURE)
    for c, p in zip(cluster_results, probs):
        c["probability"] = round(p, 4)
    cluster_results.sort(key=lambda x: x["probability"], reverse=True)
    explanation = (
        "Probabilities derived from weighted keyword sums (negations excluded) "
        f"with softmax (temperature={SOFTMAX_TEMPERATURE}). Hidden clusters < {MIN_PROB_TO_SHOW:.0%} except top-1."
    )
    return {"clusters": cluster_results, "explanation": explanation, "disclaimer": DISCLAIMER}


def format_classification(result: Dict[str, Any], top_n: int = 6, show_all: bool = False) -> str:
    lines = ["=== Symptom Pattern Estimation (Non-diagnostic) ==="]
    clusters = result["clusters"]
    if not show_all:
        # Keep top-1 always; filter remainder below MIN_PROB_TO_SHOW
        if clusters:
            top = clusters[0]
            filtered = [top] + [c for c in clusters[1:] if c["probability"] >= MIN_PROB_TO_SHOW]
            clusters = filtered
    for c in clusters[:top_n]:
        bar = '#' * max(1, int(c["probability"] * 20))
        line = f"{c['name']:<30} {c['probability']*100:5.1f}% {bar}".rstrip()
        lines.append(line)
        if c["matched_terms"]:
            lines.append(f"  features: {', '.join(c['matched_terms'])}")
    lines.append("")
    lines.append(result["explanation"])
    lines.append(result["disclaimer"])
    return "\n".join(lines)

if __name__ == "__main__":
    # Simple manual test harness
    dummy = {
        "motivo_consulta": "persistent cough and chest pain for 2 days with some shortness of breath, no fever",
        "enfermedad_actual": {
            "sintoma_principal": "cough",
            "inicio": {"expresion": "2 days", "normalizado": "P2D"},
            "caracteristicas": ["constant"]
        },
        "sintomas_asociados": ["chest pain", "shortness of breath"],
        "antecedentes_personales": ["asthma"],
        "antecedentes_familiares": [],
    }
    res = classify_case(dummy)
    print(format_classification(res))
