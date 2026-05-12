import io
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model import (
    ReductionOutput,
    find_competitors,
    generate_attributes,
    generate_insights_live,
    get_richness_config,
    normalize_names,
    reduce_dimensions,
    score_companies,
)

# -----------------------------
# APP CONFIG
# -----------------------------
st.set_page_config(page_title="Competitive Landscape Mapper", layout="wide")

_DEFAULT_KEYS = {
    "industry_in": "",
    "target_in": "",
    "competitors_in": "",
    "product_cat_in": "",
    "custom_attr_in": "",
}


def _ensure_widget_keys() -> None:
    for k, v in _DEFAULT_KEYS.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _apply_example(industry: str, target: str, competitors: str, niche: str = "") -> None:
    st.session_state["industry_in"] = industry
    st.session_state["target_in"] = target
    st.session_state["competitors_in"] = competitors
    st.session_state["product_cat_in"] = niche
    st.session_state["competitor_mode_radio"] = "Manual Input"


def _method_label(method: str) -> str:
    return {
        "pca": "PCA",
        "mds": "MDS",
        "tsne": "t-SNE",
        "umap": "UMAP",
    }.get(method, method.upper())


def _neighbor_distance_table(coords_df: pd.DataFrame, target: str, top_n: int = 6) -> pd.DataFrame:
    """Euclidean distance on the 2D map from the target brand."""
    if target not in coords_df.index or coords_df.shape[0] < 2:
        return pd.DataFrame()
    xy = coords_df[["x", "y"]].to_numpy(dtype=float)
    names = list(coords_df.index)
    ti = names.index(target)
    center = xy[ti]
    dists = np.sqrt(((xy - center) ** 2).sum(axis=1))
    order = np.argsort(dists)
    rows: list[dict[str, object]] = []
    for j in order[1 : top_n + 1]:
        rows.append({"Brand": names[int(j)], "Map distance": round(float(dists[int(j)]), 3)})
    return pd.DataFrame(rows)


def _build_perceptual_figure(
    coords_df: pd.DataFrame,
    target_company: str,
    industry: str,
    method: str,
    reduction: ReductionOutput,
) -> go.Figure:
    if target_company not in coords_df.index:
        raise ValueError("Target company missing from coordinates.")

    others = coords_df.drop(index=[target_company], errors="ignore")
    trow = coords_df.loc[target_company]

    fig = go.Figure()

    if not others.empty:
        fig.add_trace(
            go.Scatter(
                x=others["x"],
                y=others["y"],
                mode="markers+text",
                text=list(others.index),
                textposition="top center",
                name="Competitor",
                marker=dict(
                    size=17,
                    color="#5B6CFF",
                    opacity=0.92,
                    line=dict(color="#FFFFFF", width=2),
                    symbol="circle",
                ),
                textfont=dict(size=12, color="#0F172A", family="Inter, Segoe UI, system-ui, sans-serif"),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Role: Competitor<br>"
                    "x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>"
                ),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[float(trow["x"])],
            y=[float(trow["y"])],
            mode="markers+text",
            text=[target_company],
            textposition="top center",
            name="Target brand",
            marker=dict(
                size=24,
                color="#FF5A4A",
                opacity=0.95,
                line=dict(color="#FFFFFF", width=3),
                symbol="diamond",
            ),
            textfont=dict(size=13, color="#0F172A", family="Inter, Segoe UI, system-ui, sans-serif"),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Role: Target<br>"
                "x: %{x:.2f}<br>y: %{y:.2f}<extra></extra>"
            ),
        )
    )

    subtitle = f"{_method_label(method)} · {reduction.summary_line}"
    fig.update_layout(
        title=dict(
            text=f"<span style='font-size:22px'><b>{industry}</b></span><br>"
            f"<span style='font-size:13px;color:#475569'>{subtitle}</span>",
            x=0.5,
            xanchor="center",
            y=0.97,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="rgba(148,163,184,0.45)",
            borderwidth=1,
        ),
        paper_bgcolor="#F4F6FB",
        plot_bgcolor="#FAFBFE",
        margin=dict(l=28, r=28, t=120, b=72),
        font=dict(family="Inter, Segoe UI, system-ui, sans-serif", color="#0F172A", size=13),
        hoverlabel=dict(bgcolor="white", font_size=13),
        xaxis=dict(
            title=dict(text=reduction.x_axis_title, font=dict(size=13, color="#334155")),
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.35)",
            zeroline=True,
            zerolinecolor="rgba(100, 116, 139, 0.35)",
            zerolinewidth=1,
            showline=True,
            linecolor="rgba(15, 23, 42, 0.18)",
            mirror=True,
            tickfont=dict(size=11, color="#475569"),
        ),
        yaxis=dict(
            title=dict(text=reduction.y_axis_title, font=dict(size=13, color="#334155")),
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.35)",
            zeroline=True,
            zerolinecolor="rgba(100, 116, 139, 0.35)",
            zerolinewidth=1,
            showline=True,
            linecolor="rgba(15, 23, 42, 0.18)",
            mirror=True,
            tickfont=dict(size=11, color="#475569"),
            scaleanchor="x",
            scaleratio=1,
        ),
        height=700,
    )

    return fig


_ensure_widget_keys()

# -----------------------------
# SIDEBAR - Inputs & Settings
# -----------------------------
with st.sidebar:
    st.title("Settings")
    st.caption("Quick examples load realistic defaults (you can edit before running).")
    ex_cols = st.columns(3)
    with ex_cols[0]:
        if st.button("Coffee", use_container_width=True):
            _apply_example(
                "Premium Coffee",
                "Blue Bottle",
                "Starbucks, Dunkin, Peet's, Nespresso, Stumptown",
                "Third-wave specialty coffee retail",
            )
    with ex_cols[1]:
        if st.button("SaaS", use_container_width=True):
            _apply_example(
                "B2B SaaS — Product Analytics",
                "Mixpanel",
                "Amplitude, Heap, PostHog, Pendo, FullStory",
                "Digital product analytics platforms",
            )
    with ex_cols[2]:
        if st.button("CPG", use_container_width=True):
            _apply_example(
                "Plant-Based Dairy Alternatives",
                "Oatly",
                "Alpro, Califia Farms, Silk, Chobani Oat, Planet Oat",
                "Oat and almond milk beverages",
            )

    st.markdown("Enter your details to generate a strategic perceptual map.")

    industry = st.text_input("Industry", key="industry_in", placeholder="e.g., Premium Coffee")
    target_company_raw = st.text_input("Your Company Name", key="target_in", placeholder="e.g., Blue Bottle")

    st.divider()

    competitor_mode = st.radio(
        "Competitor Selection Mode",
        ["Manual Input", "AI-Generated"],
        key="competitor_mode_radio",
        help=(
            "Manual Input: You specify competitors.\n"
            "AI-Generated: The LLM discovers key competitors for you."
        ),
    )

    competitors_input = ""
    if competitor_mode == "Manual Input":
        competitors_input = st.text_area(
            "Competitors (comma-separated)",
            key="competitors_in",
            placeholder="Starbucks, Dunkin, Peet's, Nespresso",
            help="Enter your top competitors, separated by commas.",
        )
    else:
        st.info(
            "In AI-Generated mode, competitors will be discovered automatically "
            "based on your company and industry."
        )

    st.divider()

    product_category = st.text_input(
        "Product Category / Niche (Optional)",
        key="product_cat_in",
        placeholder="e.g., Shock Absorbers, Automotive Suspension Systems",
    )

    richness = st.selectbox(
        "Data Richness",
        ["Standard", "High", "Maximum"],
        help=(
            "Standard: 10 attributes, 6 competitors\n"
            "High: 20 attributes, 10 competitors\n"
            "Maximum: 30 attributes, 12 competitors"
        ),
    )

    custom_attributes = st.text_area(
        "Optional: Custom Attributes",
        key="custom_attr_in",
        placeholder="Price, Quality, Innovation (Leave blank for AI auto-gen)",
        help="Separate with commas. If left blank, the AI will determine relevant industry metrics.",
    )

    method = st.selectbox(
        "Mapping Method",
        ["pca", "mds", "tsne", "umap"],
        format_func=lambda m: {
            "pca": "PCA (variance + loadings)",
            "mds": "MDS (distance preservation)",
            "tsne": "t-SNE (local clusters)",
            "umap": "UMAP (structure + separation)",
        }[m],
        help="PCA: interpretable axes and variance. MDS/t-SNE/UMAP: alternative geometry for positioning.",
    )

    run_button = st.button("Generate Landscape", type="primary", use_container_width=True)

# -----------------------------
# MAIN UI - Results Area
# -----------------------------
st.title("Competitive Landscape Mapping")

if not run_button:
    st.info("Fill in the sidebar and click **Generate Landscape** to start.")
    st.markdown(
        """
        ### How it works
        1. **LLM synthesis:** GPT-4o-mini proposes attributes and scores brands (chunked for reliability).
        2. **Reduction:** PCA, MDS, t-SNE, or UMAP compress scores into 2D coordinates.
        3. **Outputs:** Interactive map, strategic insights, and downloadable data and chart files.
        """
    )

if run_button:
    if not target_company_raw or not industry:
        st.error("Please provide a Company Name and Industry.")
    elif competitor_mode == "Manual Input" and not str(competitors_input).strip():
        st.error("In Manual Input mode, please provide at least one competitor.")
    else:
        target_company = target_company_raw.strip().title()
        n_attributes, n_competitors = get_richness_config(richness)  # type: ignore

        try:
            with st.status("Building your market map...", expanded=True) as status:
                st.write("Identifying competitors...")

                if competitor_mode == "Manual Input":
                    competitors_raw = [c for c in str(competitors_input).split(",") if c.strip()]
                    competitors = normalize_names(competitors_raw)
                else:
                    niche = product_category.strip() if product_category else ""
                    pc = niche if niche else industry
                    competitors = find_competitors(
                        company=target_company,
                        industry=industry,
                        product_category=pc,
                        n_competitors=n_competitors,
                    )

                if not competitors:
                    st.error("No competitors could be determined. Try Manual mode or adjust inputs.")
                    status.update(label="Failed to determine competitors.", state="error", expanded=True)
                    st.stop()

                competitors = [c for c in competitors if c != target_company]
                competitors = competitors[:n_competitors]
                all_companies = [target_company] + competitors

                st.write("Determining competitive attributes...")
                if str(custom_attributes).strip():
                    attributes = [
                        a.strip().title() for a in str(custom_attributes).split(",") if a.strip()
                    ]
                else:
                    attributes = generate_attributes(industry=industry, n_attributes=n_attributes)

                if not attributes:
                    st.error("No attributes could be determined. Add custom attributes or try again.")
                    status.update(label="Failed to determine attributes.", state="error", expanded=True)
                    st.stop()

                st.write("Scoring brands (batched for stable JSON)...")
                feature_df = score_companies(all_companies, attributes, industry)

                st.write("Computing 2D coordinates...")
                reduction: ReductionOutput = reduce_dimensions(feature_df, method=method)  # type: ignore

                st.write("Generating strategic insights...")
                insights = generate_insights_live(
                    reduction.coords_df,
                    target_company,
                    feature_df,
                    reduction.summary_line,
                )

                status.update(label="Analysis complete", state="complete", expanded=False)

            coords_df = reduction.coords_df

            st.subheader("1. Perceptual map")
            fig = _build_perceptual_figure(
                coords_df,
                target_company=target_company,
                industry=industry,
                method=method,
                reduction=reduction,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Reading the map")
            read_l, read_r = st.columns((1.35, 1.0))
            with read_l:
                with st.container(border=True):
                    st.markdown(reduction.axis_explanation_md)
            with read_r:
                with st.container(border=True):
                    st.markdown("**Closest peers on the map**")
                    st.caption("Straight-line distance in the 2D layout (not geographic).")
                    nn = _neighbor_distance_table(coords_df, target_company, top_n=6)
                    if nn.empty:
                        st.caption("Add at least one competitor to see peer distances.")
                    else:
                        st.dataframe(nn, use_container_width=True, hide_index=True)

            st.subheader("2. Strategic insights")
            with st.container(border=True):
                st.markdown(insights)

            st.divider()
            st.subheader("3. Exports")
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
            base = f"landscape-{ts}"

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.download_button(
                    label="Scores CSV",
                    data=feature_df.to_csv().encode("utf-8"),
                    file_name=f"{base}-scores.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with c2:
                coords_out = coords_df.copy()
                st.download_button(
                    label="Coordinates CSV",
                    data=coords_out.to_csv().encode("utf-8"),
                    file_name=f"{base}-coords.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with c3:
                report = (
                    f"# Competitive landscape — {industry}\n\n"
                    f"- **Target:** {target_company}\n"
                    f"- **Method:** {method.upper()} — {reduction.summary_line}\n"
                    f"- **Brands:** {', '.join(all_companies)}\n\n"
                    f"## How to read the map\n\n{reduction.axis_explanation_md}\n\n"
                    f"## Insights\n\n{insights}\n"
                )
                st.download_button(
                    label="Report (.md)",
                    data=report.encode("utf-8"),
                    file_name=f"{base}-report.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with c4:
                buf = io.StringIO()
                fig.write_html(buf, include_plotlyjs="cdn", full_html=True)
                st.download_button(
                    label="Chart HTML",
                    data=buf.getvalue().encode("utf-8"),
                    file_name=f"{base}-map.html",
                    mime="text/html",
                    use_container_width=True,
                )

            png_row = st.columns(2)
            with png_row[0]:
                try:
                    png_bytes = fig.to_image(format="png", width=1400, height=900, scale=1)
                    st.download_button(
                        label="Chart PNG",
                        data=png_bytes,
                        file_name=f"{base}-map.png",
                        mime="image/png",
                        use_container_width=True,
                    )
                except Exception:
                    st.caption("PNG export needs Kaleido on the server; HTML download always works.")

            with png_row[1]:
                try:
                    svg_bytes = fig.to_image(format="svg", width=1400, height=900, scale=1)
                    st.download_button(
                        label="Chart SVG",
                        data=svg_bytes,
                        file_name=f"{base}-map.svg",
                        mime="image/svg+xml",
                        use_container_width=True,
                    )
                except Exception:
                    pass

            st.divider()
            with st.expander("Detailed attribute scores (1–10)"):
                st.dataframe(feature_df.style.background_gradient(cmap="Blues", axis=None))

        except Exception as e:
            st.error(f"Something went wrong: {str(e)}")
