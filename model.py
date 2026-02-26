import os
import json
from typing import List, Dict, Literal, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler

# Load environment variables
load_dotenv()

# Load API key (Streamlit Cloud → st.secrets, Local → .env)
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise ValueError(
        "OPENAI_API_KEY not found. Add it to .env (local) or st.secrets (Streamlit Cloud)."
    )

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# LLM HELPERS
# -----------------------------
def _call_llm_json(system_prompt: str, user_prompt: str) -> Dict:
    """
    Forces the model to return a valid JSON object.
    Uses gpt-4o-mini for cost efficiency.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt + " Respond only in valid JSON."},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    return json.loads(response.choices[0].message.content)

# -----------------------------
# ATTRIBUTE GENERATION
# -----------------------------
def generate_attributes(industry: str, n_attributes: int = 10) -> List[str]:
    system_prompt = "You are a marketing analytics assistant."
    user_prompt = (
        f"Generate {n_attributes} distinct attributes for competitive positioning "
        f"in the '{industry}' industry. Return a JSON object with a key 'attributes' "
        "containing a list of strings."
    )

    try:
        data = _call_llm_json(system_prompt, user_prompt)
        return [str(a).strip() for a in data.get("attributes", [])][:n_attributes]
    except Exception:
        return ["Quality", "Price", "Innovation", "Customer Service", "Brand Loyalty"]

# -----------------------------
# COMPANY SCORING
# -----------------------------
def score_companies(companies: List[str], attributes: List[str], industry: str) -> pd.DataFrame:
    system_prompt = (
        "You are a marketing analyst. Rate companies 1-10 on attributes. "
        "Output a JSON object where keys are company names."
    )
    user_prompt = (
        f"Industry: {industry}\nCompanies: {companies}\nAttributes: {attributes}\n"
        "Assign integer scores (1-10) for each attribute based on market perception."
    )

    data = _call_llm_json(system_prompt, user_prompt)

    df = pd.DataFrame.from_dict(data, orient="index")
    df = df.reindex(index=companies, columns=attributes).fillna(5.0)
    return df.astype(float)

# -----------------------------
# DIMENSIONALITY REDUCTION
# -----------------------------
def reduce_dimensions(
    feature_df: pd.DataFrame,
    method: Literal["pca", "mds"] = "pca",
    random_state: int = 42,
) -> pd.DataFrame:

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
def generate_insights(coords_df: pd.DataFrame, target_company: str, feature_df: pd.DataFrame) -> str:
    coords_dict = coords_df.round(2).to_dict(orient="index")
    feature_summary = feature_df.round(1).to_dict(orient="index")

    system_prompt = "You are a senior marketing strategist. Provide concise competitive insights."
    user_prompt = (
        f"Target: {target_company}\nMap Coordinates: {coords_dict}\n"
        f"Raw Scores: {feature_summary}\n\n"
        "Explain: 1) Nearest competitors 2) Market position 3) Market gaps (white space)."
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
