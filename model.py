import os
import json
from typing import List, Dict, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler

# -----------------------------
# ENV & CLIENT SETUP
# -----------------------------
load_dotenv()

# Prefer Streamlit secrets (Cloud), fall back to .env (local)
OPENAI_API_KEY: Optional[str] = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise ValueError(
        "OPENAI_API_KEY not found. Add it to .env (local) or st.secrets (Streamlit Cloud)."
    )

client = OpenAI(api_key=OPENAI_API_KEY)


# -----------------------------
# RICHNESS CONFIG
# -----------------------------
RichnessLevel = Literal["Standard", "High", "Maximum"]


def get_richness_config(richness: RichnessLevel) -> Tuple[int, int]:
    """
    Map richness level to (n_attributes, n_competitors).
    - Standard: 10 attributes, 6 competitors
    - High: 20 attributes, 10 competitors
    - Maximum: 30 attributes, 12 competitors
    """
    if richness == "High":
        return 20, 10
    if richness == "Maximum":
        return 30, 12
    # Default: Standard
    return 10, 6


# -----------------------------
# LLM HELPERS
# -----------------------------
def _call_llm_json(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
) -> Dict:
    """
    Call the LLM and force a valid JSON object response.
    Uses gpt-4o-mini for cost efficiency.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": system_prompt + " Respond only in valid JSON.",
            },
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
    )
    try:
        return json.loads(response.choices[0].message.content)
    except Exception:
        return {}


# -----------------------------
# NORMALIZATION HELPERS
# -----------------------------
def normalize_names(names: List[str]) -> List[str]:
    """
    Normalize company names to a consistent format (case-insensitive, trimmed).
    Also deduplicates while preserving order.
    """
    seen = set()
    normalized: List[str] = []
    for name in names:
        clean = name.strip()
        if not clean:
            continue
        # Title-case for consistency (Starbucks, Qualtrics, etc.)
        clean = clean.title()
        if clean not in seen:
            seen.add(clean)
            normalized.append(clean)
    return normalized


# -----------------------------
# ATTRIBUTE GENERATION
# -----------------------------
def generate_attributes(
    industry: str,
    n_attributes: int = 25,
) -> List[str]:
    """
    Generate a rich set of competitive positioning attributes for a given industry.
    Defaults to a higher number of attributes to maximize signal.
    """
    system_prompt = "You are a senior marketing analytics assistant."
    user_prompt = (
        f"Generate {n_attributes} distinct, meaningful attributes for competitive positioning "
        f"in the '{industry}' industry. These should reflect how customers and the market "
        "perceive brands (e.g., price, quality, innovation, support, UX, integrations, etc.). "
        "Return a JSON object with a key 'attributes' containing a list of short attribute names. "
        "Avoid duplicates and overly generic placeholders."
    )

    data = _call_llm_json(system_prompt, user_prompt, temperature=0.3)
    attrs = [str(a).strip() for a in data.get("attributes", []) if str(a).strip()]

    # Fallback if LLM fails or returns too few
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

    # Normalize attribute names a bit
    attrs = [a.strip().title() for a in attrs]
    return attrs[:n_attributes]


# -----------------------------
# AI COMPETITOR DISCOVERY
# -----------------------------
def find_competitors(
    company: str,
    industry: str,
    n_competitors: int = 10,
) -> List[str]:
    """
    Use the LLM to discover key competitors for a given company and industry.
    """
    system_prompt = "You are a market research assistant."
    user_prompt = (
        f"Identify {n_competitors} major direct competitors for the company '{company}' "
        f"in the '{industry}' industry. Focus on realistic, well-known or highly relevant players. "
        "Return a JSON object with a key 'competitors' containing a list of company names."
    )

    data = _call_llm_json(system_prompt, user_prompt, temperature=0.3)
    competitors = [str(c).strip() for c in data.get("competitors", []) if str(c).strip()]

    return normalize_names(competitors)[:n_competitors]


# -----------------------------
# COMPANY SCORING
# -----------------------------
def score_companies(
    companies: List[str],
    attributes: List[str],
    industry: str,
) -> pd.DataFrame:
    """
    Ask the LLM to score each company on each attribute (1-10).
    Uses a prompt that encourages full use of the scale and avoids flat 5/5 scoring.
    """
    system_prompt = (
        "You are a senior marketing analyst. Rate companies from 1 to 10 on each attribute "
        "based on realistic market perception. Use the full range of the scale where appropriate. "
        "Do not give all companies the same score on all attributes. "
        "Output a JSON object where keys are company names and values are objects "
        "with attribute: score pairs."
    )

    user_prompt = (
        f"Industry: {industry}\n"
        f"Companies: {companies}\n"
        f"Attributes: {attributes}\n\n"
        "Assign integer scores from 1 to 10 for each attribute for each company. "
        "Be realistic and differentiate companies based on how they are generally perceived "
        "in this industry."
    )

    data = _call_llm_json(system_prompt, user_prompt, temperature=0.4)

    # Build DataFrame, enforce structure
    df = pd.DataFrame.from_dict(data, orient="index")

    # Ensure all companies and attributes are present
    df = df.reindex(index=companies, columns=attributes)

    # Fill missing values with a neutral mid-score
    df = df.fillna(5.0)

    # Ensure numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(5.0)

    return df.astype(float)


# -----------------------------
# DIMENSIONALITY REDUCTION
# -----------------------------
def reduce_dimensions(
    feature_df: pd.DataFrame,
    method: Literal["pca", "mds"] = "pca",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Reduce high-dimensional attribute space to 2D coordinates using PCA or MDS.
    """
    X_scaled = StandardScaler().fit_transform(feature_df.values)

    if method == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
    else:
        reducer = MDS(n_components=2, random_state=random_state)

    coords = reducer.fit_transform(X_scaled)
    return pd.DataFrame(coords, index=feature_df.index, columns=["x", "y"])


# -----------------------------
# INSIGHT GENERATION
# -----------------------------
def generate_insights(
    coords_df: pd.DataFrame,
    target_company: str,
    feature_df: pd.DataFrame,
) -> str:
    """
    Generate strategic insights from the 2D map and raw scores.
    """
    coords_dict = coords_df.round(2).to_dict(orient="index")
    feature_summary = feature_df.round(1).to_dict(orient="index")

    system_prompt = (
        "You are a senior marketing strategist. Provide concise, insightful competitive analysis "
        "based on a perceptual map and underlying attribute scores."
    )

    user_prompt = (
        f"Target company: {target_company}\n\n"
        f"Map coordinates (2D positioning):\n{coords_dict}\n\n"
        f"Raw attribute scores (1-10):\n{feature_summary}\n\n"
        "Explain in clear bullet-style paragraphs:\n"
        "1) Nearest competitors and how they compare.\n"
        "2) The target company's market position and differentiation.\n"
        "3) Market gaps or 'white space' opportunities the target could pursue.\n"
        "Be specific and actionable, but concise."
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
    )

    return response.choices[0].message.content.strip()
