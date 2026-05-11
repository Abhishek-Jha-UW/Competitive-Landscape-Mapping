import os
import json
import time
from io import StringIO
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.preprocessing import StandardScaler

# -----------------------------
# ENV & CLIENT SETUP
# -----------------------------
load_dotenv()

OPENAI_API_KEY: Optional[str] = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise ValueError(
        "OPENAI_API_KEY not found. Add it to .env (local) or st.secrets (Streamlit Cloud)."
    )

client = OpenAI(api_key=OPENAI_API_KEY)

LLM_MODEL = "gpt-4o-mini"
MAX_JSON_RETRIES = 3
SCORE_CHUNK_SIZE = 4


# -----------------------------
# RICHNESS CONFIG
# -----------------------------
RichnessLevel = Literal["Standard", "High", "Maximum"]
ReductionMethod = Literal["pca", "mds", "tsne", "umap"]


def get_richness_config(richness: RichnessLevel) -> Tuple[int, int]:
    if richness == "High":
        return 20, 10
    if richness == "Maximum":
        return 30, 12
    return 10, 6


# -----------------------------
# REDUCTION OUTPUT
# -----------------------------
@dataclass(frozen=True)
class ReductionOutput:
    coords_df: pd.DataFrame
    x_axis_title: str
    y_axis_title: str
    summary_line: str


# -----------------------------
# LLM HELPERS
# -----------------------------
def _safe_json_loads(text: Optional[str]) -> Dict[str, Any]:
    if not text or not str(text).strip():
        return {}
    try:
        out = json.loads(text)
        return out if isinstance(out, dict) else {}
    except json.JSONDecodeError:
        return {}


def _call_llm_raw(
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    json_mode: bool,
) -> str:
    kwargs: Dict[str, Any] = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    msg = response.choices[0].message
    return (msg.content or "").strip()


def _call_llm_json(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_retries: int = MAX_JSON_RETRIES,
) -> Dict[str, Any]:
    """
    Call the LLM with json_object mode, retries, and a single repair pass on failure.
    """
    sys_json = system_prompt + " Respond only with a single valid JSON object (no markdown, no prose)."
    last_raw = ""

    for attempt in range(max_retries):
        try:
            last_raw = _call_llm_raw(sys_json, user_prompt, temperature, json_mode=True)
            data = _safe_json_loads(last_raw)
            if data:
                return data
        except Exception:
            pass
        time.sleep(0.4 * (attempt + 1))

    repair_user = (
        "Your previous output was not usable JSON or was empty. "
        "Return ONLY a valid JSON object that satisfies the original instructions.\n\n"
        f"Original instructions:\n{user_prompt}\n\n"
        f"Previous output (may be invalid):\n{last_raw[:12000]}"
    )
    try:
        last_raw = _call_llm_raw(
            "You output strict JSON only. No keys outside the requested schema.",
            repair_user,
            min(0.1, temperature),
            json_mode=True,
        )
        data = _safe_json_loads(last_raw)
        if data:
            return data
    except Exception:
        pass

    return {}


# -----------------------------
# NORMALIZATION HELPERS
# -----------------------------
def normalize_names(names: List[str]) -> List[str]:
    seen = set()
    normalized: List[str] = []
    for name in names:
        clean = name.strip()
        if not clean:
            continue
        clean = clean.title()
        if clean not in seen:
            seen.add(clean)
            normalized.append(clean)
    return normalized


def _axis_label_from_loadings(
    component: np.ndarray,
    attr_names: List[str],
    prefix: str,
    top_k: int = 2,
) -> str:
    """Short label: top positive and negative loadings for one PC."""
    if component.size == 0 or not attr_names:
        return prefix
    order = np.argsort(np.abs(component))[::-1]
    picks: List[str] = []
    for idx in order[: max(top_k * 3, 6)]:
        if len(picks) >= top_k * 2:
            break
        a = attr_names[int(idx)]
        w = float(component[idx])
        if abs(w) < 1e-6:
            continue
        sign = "+" if w >= 0 else "-"
        picks.append(f"{a} ({sign})")
    if not picks:
        return prefix
    label = ", ".join(picks[:4])
    return f"{prefix}: {label}"


# -----------------------------
# ATTRIBUTE GENERATION (uncached core)
# -----------------------------
def _generate_attributes_core(industry: str, n_attributes: int) -> List[str]:
    system_prompt = "You are a senior marketing analytics assistant."
    user_prompt = (
        f"Generate {n_attributes} distinct, meaningful attributes for competitive positioning "
        f"in the '{industry}' industry. These should reflect how customers and the market "
        "perceive brands (e.g., price, quality, innovation, support, UX, integrations). "
        "Return a JSON object with key 'attributes' containing a list of short attribute names. "
        "Avoid duplicates and vague placeholders."
    )
    data = _call_llm_json(system_prompt, user_prompt, temperature=0.3)
    attrs = [str(a).strip() for a in data.get("attributes", []) if str(a).strip()]

    if len(attrs) < max(5, n_attributes // 2):
        fallback = [
            "Price",
            "Quality",
            "Innovation",
            "Customer Service",
            "Brand Awareness",
            "Ease Of Use",
            "Feature Depth",
            "Integration Capabilities",
            "Scalability",
            "Reliability",
            "Security",
            "Customization",
            "Support Responsiveness",
            "User Experience",
            "Analytics & Reporting",
            "Implementation Speed",
            "Total Cost Of Ownership",
            "Ecosystem",
            "Reputation",
            "Flexibility",
        ]
        attrs = fallback[:n_attributes]

    attrs = [a.strip().title() for a in attrs]
    return attrs[:n_attributes]


@st.cache_data(ttl="6h", show_spinner=False)
def generate_attributes(industry: str, n_attributes: int) -> List[str]:
    return _generate_attributes_core(industry, n_attributes)


# -----------------------------
# AI COMPETITOR DISCOVERY
# -----------------------------
def _find_competitors_core(
    company: str,
    industry: str,
    product_category: str,
    n_competitors: int,
) -> List[str]:
    system_prompt = "You are a market research analyst specializing in niche competitor identification."
    user_prompt = (
        f"Identify {n_competitors} direct competitors for the company '{company}'.\n"
        f"Industry: {industry}\n"
        f"Product Category / Niche: {product_category}\n\n"
        "Focus ONLY on companies that manufacture similar products or operate in the same niche.\n"
        "Avoid large general conglomerates unless they directly compete in this product category.\n"
        "Return a JSON object with key 'competitors' containing a list of company names."
    )
    data = _call_llm_json(system_prompt, user_prompt, temperature=0.3)
    competitors = [str(c).strip() for c in data.get("competitors", []) if str(c).strip()]
    return normalize_names(competitors)[:n_competitors]


@st.cache_data(ttl="6h", show_spinner=False)
def find_competitors(
    company: str,
    industry: str,
    product_category: str,
    n_competitors: int,
) -> List[str]:
    return _find_competitors_core(company, industry, product_category, n_competitors)


# -----------------------------
# COMPANY SCORING
# -----------------------------
def _score_companies_one_batch(
    companies: Tuple[str, ...],
    attributes: Tuple[str, ...],
    industry: str,
) -> pd.DataFrame:
    company_list = list(companies)
    attr_list = list(attributes)
    system_prompt = (
        "You are a senior marketing analyst. Rate companies from 1 to 10 on each attribute "
        "based on realistic market perception. Use the full scale where appropriate. "
        "Do not give all companies the same score on every attribute. "
        "Output a JSON object whose keys are EXACT company names from the user list, "
        "and each value is an object mapping each EXACT attribute name to an integer 1–10."
    )
    user_prompt = (
        f"Industry: {industry}\n"
        f"Companies (use these exact keys): {company_list}\n"
        f"Attributes (use these exact keys): {attr_list}\n\n"
        "Return scores for every company for every attribute."
    )
    data = _call_llm_json(system_prompt, user_prompt, temperature=0.35)
    df = pd.DataFrame.from_dict(data, orient="index")
    df = df.reindex(index=company_list, columns=attr_list)
    df = df.fillna(5.0)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(5.0)
    return df.astype(float)


@st.cache_data(ttl="6h", show_spinner=False)
def _score_companies_cached(
    companies: Tuple[str, ...],
    attributes: Tuple[str, ...],
    industry: str,
) -> pd.DataFrame:
    """
    Cached scoring. Large brand sets are scored in chunks to keep JSON payloads reliable.
    """
    if not companies or not attributes:
        return pd.DataFrame()

    if len(companies) <= SCORE_CHUNK_SIZE:
        return _score_companies_one_batch(companies, attributes, industry)

    parts: List[pd.DataFrame] = []
    for i in range(0, len(companies), SCORE_CHUNK_SIZE):
        chunk = companies[i : i + SCORE_CHUNK_SIZE]
        parts.append(_score_companies_one_batch(chunk, attributes, industry))

    merged = pd.concat(parts)
    merged = merged.reindex(index=list(companies), columns=list(attributes))
    merged = merged.fillna(5.0)
    return merged.astype(float)


def score_companies(
    companies: List[str],
    attributes: List[str],
    industry: str,
) -> pd.DataFrame:
    return _score_companies_cached(tuple(companies), tuple(attributes), industry)


# -----------------------------
# DIMENSIONALITY REDUCTION
# -----------------------------
def reduce_dimensions(
    feature_df: pd.DataFrame,
    method: ReductionMethod = "pca",
    random_state: int = 42,
) -> ReductionOutput:
    """
    Reduce attribute space to 2D. PCA includes variance explained and loading-based axis titles.
    """
    if feature_df.shape[0] < 2:
        raise ValueError("Need at least two brands to build a map.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df.values)
    attr_names = list(feature_df.columns)
    n_samples = X_scaled.shape[0]

    if method == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
        coords = reducer.fit_transform(X_scaled)
        ev = reducer.explained_variance_ratio_
        pct = [100.0 * float(ev[0]), 100.0 * float(ev[1])]
        comp = reducer.components_
        x_lab = _axis_label_from_loadings(comp[0], attr_names, "PC1")
        y_lab = _axis_label_from_loadings(comp[1], attr_names, "PC2")
        summary = f"PCA — PC1 {pct[0]:.1f}% · PC2 {pct[1]:.1f}% variance (standardized inputs)"
        coords_df = pd.DataFrame(coords, index=feature_df.index, columns=["x", "y"])
        return ReductionOutput(coords_df, x_lab, y_lab, summary)

    if method == "mds":
        reducer = MDS(n_components=2, random_state=random_state)
        coords = reducer.fit_transform(X_scaled)
        summary = "MDS (metric) on standardized attributes — distances reflect dissimilarity"
        coords_df = pd.DataFrame(coords, index=feature_df.index, columns=["x", "y"])
        return ReductionOutput(coords_df, "Dimension 1", "Dimension 2", summary)

    if method == "tsne":
        perplexity = float(max(1, min(30, n_samples - 1)))
        tsne_kwargs = dict(
            n_components=2,
            random_state=random_state,
            perplexity=perplexity,
            init="pca",
        )
        try:
            reducer = TSNE(learning_rate="auto", **tsne_kwargs)  # type: ignore[arg-type]
        except TypeError:
            reducer = TSNE(learning_rate=200, **tsne_kwargs)
        coords = reducer.fit_transform(X_scaled)
        summary = f"t-SNE (perplexity={perplexity:.0f}) — local neighborhoods on standardized attributes"
        coords_df = pd.DataFrame(coords, index=feature_df.index, columns=["x", "y"])
        return ReductionOutput(coords_df, "t-SNE 1", "t-SNE 2", summary)

    if method == "umap":
        try:
            import umap  # type: ignore
        except ImportError as e:
            raise ImportError(
                "umap-learn is required for UMAP. Add 'umap-learn' to requirements and redeploy."
            ) from e
        n_neighbors = max(1, min(15, n_samples - 1))
        reducer = umap.UMAP(
            n_components=2,
            random_state=random_state,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric="euclidean",
        )
        coords = reducer.fit_transform(X_scaled)
        summary = f"UMAP (n_neighbors={n_neighbors}) on standardized attributes"
        coords_df = pd.DataFrame(coords, index=feature_df.index, columns=["x", "y"])
        return ReductionOutput(coords_df, "UMAP 1", "UMAP 2", summary)

    raise ValueError(f"Unknown method: {method}")


# -----------------------------
# INSIGHT GENERATION
# -----------------------------
@st.cache_data(ttl="6h", show_spinner=False)
def generate_insights(
    coords_csv: str,
    target_company: str,
    feature_csv: str,
    map_summary: str,
) -> str:
    coords_df = pd.read_csv(StringIO(coords_csv), index_col=0)
    feature_df = pd.read_csv(StringIO(feature_csv), index_col=0)

    coords_dict = coords_df.round(2).to_dict(orient="index")
    feature_summary = feature_df.round(1).to_dict(orient="index")

    system_prompt = (
        "You are a senior marketing strategist. Provide concise, insightful competitive analysis "
        "based on a perceptual map and underlying attribute scores."
    )
    user_prompt = (
        f"Target company: {target_company}\n"
        f"Map method note: {map_summary}\n\n"
        f"Map coordinates (2D positioning):\n{coords_dict}\n\n"
        f"Raw attribute scores (1-10):\n{feature_summary}\n\n"
        "Explain in clear bullet-style paragraphs:\n"
        "1) Nearest competitors and how they compare.\n"
        "2) The target company's market position and differentiation.\n"
        "3) Market gaps or 'white space' opportunities the target could pursue.\n"
        "Be specific and actionable, but concise."
    )

    text = _call_llm_raw(system_prompt, user_prompt, temperature=0.4, json_mode=False)
    return (text or "").strip()


def generate_insights_live(
    coords_df: pd.DataFrame,
    target_company: str,
    feature_df: pd.DataFrame,
    map_summary: str,
) -> str:
    """Serialize frames for cache key stability."""
    buf_c = pd.io.common.StringIO()
    buf_f = pd.io.common.StringIO()
    coords_df.to_csv(buf_c)
    feature_df.to_csv(buf_f)
    return generate_insights(buf_c.getvalue(), target_company, buf_f.getvalue(), map_summary)
