import io
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
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

            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader("1. Perceptual map")
                st.caption(reduction.summary_line)

                coords_plot_df = coords_df.copy()
                coords_plot_df["Type"] = [
                    "Target Brand" if i == target_company else "Competitor" for i in coords_plot_df.index
                ]

                fig = px.scatter(
                    coords_plot_df,
                    x="x",
                    y="y",
                    text=coords_plot_df.index,
                    color="Type",
                    color_discrete_map={"Target Brand": "#EF553B", "Competitor": "#636EFA"},
                    title=f"Market positioning — {industry}",
                    template="plotly_white",
                    height=620,
                    labels={"x": reduction.x_axis_title, "y": reduction.y_axis_title},
                )
                fig.update_traces(
                    textposition="top center",
                    marker=dict(size=14, line=dict(width=2, color="DarkSlateGrey")),
                )
                fig.update_layout(xaxis_title=reduction.x_axis_title, yaxis_title=reduction.y_axis_title)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("2. Strategic insights")
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
                    png_bytes = fig.to_image(format="png", width=1200, height=800, scale=1)
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
                    svg_bytes = fig.to_image(format="svg", width=1200, height=800, scale=1)
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
