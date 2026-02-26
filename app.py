import streamlit as st
import pandas as pd
import plotly.express as px

from model import (
    generate_attributes,
    score_companies,
    reduce_dimensions,
    generate_insights,
    find_competitors,
    normalize_names,
    get_richness_config,
)

# -----------------------------
# APP CONFIG
# -----------------------------
st.set_page_config(page_title="Competitive Landscape Mapper", layout="wide")

# -----------------------------
# SIDEBAR - Inputs & Settings
# -----------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("Enter your details to generate a strategic perceptual map.")

    industry = st.text_input("Industry", placeholder="e.g., Premium Coffee")
    target_company_raw = st.text_input("Your Company Name", placeholder="e.g., Blue Bottle")

    st.divider()

    competitor_mode = st.radio(
        "Competitor Selection Mode",
        ["Manual Input", "AI-Generated"],
        help=(
            "Manual Input: You specify competitors.\n"
            "AI-Generated: The LLM discovers key competitors for you."
        ),
    )

    competitors_input = ""
    if competitor_mode == "Manual Input":
        competitors_input = st.text_area(
            "Competitors (comma-separated)",
            placeholder="Starbucks, Dunkin, Peet's, Nespresso",
            help="Enter your top competitors, separated by commas.",
        )
    else:
        st.info(
            "In AI-Generated mode, competitors will be discovered automatically "
            "based on your company and industry."
        )

    st.divider()

    # OPTIONAL niche field
    product_category = st.text_input(
        "Product Category / Niche (Optional)",
        placeholder="e.g., Shock Absorbers, Automotive Suspension Systems"
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
        placeholder="Price, Quality, Innovation (Leave blank for AI auto-gen)",
        help="Separate with commas. If left blank, the AI will determine relevant industry metrics.",
    )

    method = st.selectbox(
        "Mapping Method",
        ["pca", "mds"],
        help="PCA highlights variance; MDS highlights relative similarity.",
    )

    run_button = st.button("Generate Landscape", type="primary", use_container_width=True)

# -----------------------------
# MAIN UI - Results Area
# -----------------------------
st.title("📊 Competitive Landscape Mapping Tool")

if not run_button:
    st.info("👈 Fill in the details in the sidebar and click 'Generate Landscape' to start.")
    st.markdown(
        """
        ### How it works:
        1. **LLM Synthesis:** The tool uses GPT-4o-mini to research and score brands.
        2. **Dimensionality Reduction:** We compress high-dimensional scores into 2D space.
        3. **Market Mapping:** We visualize the 'White Space' where you can differentiate.
        """
    )

if run_button:
    # Basic validation
    if not target_company_raw or not industry:
        st.error("Please provide a Company Name and Industry.")
    elif competitor_mode == "Manual Input" and not competitors_input.strip():
        st.error("In Manual Input mode, please provide at least one competitor.")
    else:
        # Normalize target company name
        target_company = target_company_raw.strip().title()

        # Get richness config
        n_attributes, n_competitors = get_richness_config(richness)  # type: ignore
        
        try:
            with st.status("🏗️ Building your market map...", expanded=True) as status:
                # 1. Competitors
                st.write("🏢 Identifying competitors...")

                if competitor_mode == "Manual Input":
                    competitors_raw = [c for c in competitors_input.split(",") if c.strip()]
                    competitors = normalize_names(competitors_raw)
                else:
                    # If product category is provided → niche-specific competitor discovery
                    if product_category.strip():
                        competitors = find_competitors(
                            company=target_company,
                            industry=industry,
                            product_category=product_category,
                            n_competitors=n_competitors,
                        )
                    else:
                        # Fallback: general competitor discovery
                        competitors = find_competitors(
                            company=target_company,
                            industry=industry,
                            product_category=industry,  # fallback to industry itself
                            n_competitors=n_competitors,
                        )

                if not competitors:
                    st.error("No competitors could be determined. Please try Manual mode or adjust inputs.")
                    status.update(label="Failed to determine competitors.", state="error", expanded=True)
                    st.stop()

                # Ensure target is not duplicated in competitors
                competitors = [c for c in competitors if c != target_company]

                # Limit competitors to richness config
                competitors = competitors[:n_competitors]
                all_companies = [target_company] + competitors

                # 2. Attributes
                st.write("🔍 Determining competitive attributes...")
                if custom_attributes.strip():
                    attributes = [a.strip().title() for a in custom_attributes.split(",") if a.strip()]
                else:
                    attributes = generate_attributes(industry=industry, n_attributes=n_attributes)

                if not attributes:
                    st.error("No attributes could be determined. Please provide custom attributes or try again.")
                    status.update(label="Failed to determine attributes.", state="error", expanded=True)
                    st.stop()

                # 3. Score
                st.write("📊 Scoring brands on perception...")
                feature_df = score_companies(all_companies, attributes, industry)

                # 4. Reduce
                st.write("🗺️ Calculating 2D coordinates...")
                coords_df = reduce_dimensions(feature_df, method=method)

                # 5. Insights
                st.write("💡 Generating strategic insights...")
                insights = generate_insights(coords_df, target_company, feature_df)

                status.update(label="Analysis Complete!", state="complete", expanded=False)

            # --- Layout: Map & Insights ---
            col1, col2 = st.columns([2, 1])

            with col1:
                st.header("1. Perceptual Map")

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
                    title=f"Market Positioning: {industry}",
                    template="plotly_white",
                    height=600,
                )
                fig.update_traces(
                    textposition="top center",
                    marker=dict(size=14, line=dict(width=2, color="DarkSlateGrey")),
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.header("2. Strategic Insights")
                st.markdown(insights)

            # --- Data Breakdown ---
            st.divider()
            with st.expander("View Detailed Attribute Scores (1-10)"):
                st.dataframe(feature_df.style.background_gradient(cmap="Blues", axis=1))

        except Exception as e:
            st.error(f"Something went wrong: {str(e)}")
