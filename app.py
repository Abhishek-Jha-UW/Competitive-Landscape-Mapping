import streamlit as st
import pandas as pd
import plotly.express as px
from model import (
    generate_attributes, 
    score_companies, 
    reduce_dimensions, 
    generate_insights
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
    target_company = st.text_input("Your Company Name", placeholder="e.g., Blue Bottle")
    competitors_input = st.text_area(
        "Competitors (comma-separated)", 
        placeholder="Starbucks, Dunkin, Peet's, Nespresso"
    )
    
    st.divider()
    
    custom_attributes = st.text_area(
        "Optional: Custom Attributes",
        placeholder="Price, Quality, Innovation (Leave blank for AI auto-gen)",
        help="Separate with commas. If left blank, the AI will determine relevant industry metrics."
    )
    
    method = st.selectbox(
        "Mapping Method", 
        ["pca", "mds"], 
        help="PCA highlights variance; MDS highlights relative similarity."
    )
    
    run_button = st.button("Generate Landscape", type="primary", use_container_width=True)

# -----------------------------
# MAIN UI - Results Area
# -----------------------------
st.title("📊 Competitive Landscape Mapping Tool")

if not run_button:
    st.info("👈 Fill in the details in the sidebar and click 'Generate Landscape' to start.")
    st.markdown("""
    ### How it works:
    1. **LLM Synthesis:** The tool uses GPT-4o-mini to research and score brands.
    2. **Dimensionality Reduction:** We compress high-dimensional scores into 2D space.
    3. **Market Mapping:** We visualize the 'White Space' where you can differentiate.
    """)

if run_button:
    if not target_company or not competitors_input or not industry:
        st.error("Please provide a Company Name, Industry, and at least one Competitor.")
    else:
        # Process inputs
        competitors = [c.strip() for c in competitors_input.split(",") if c.strip()]
        all_companies = [target_company] + competitors
        
        try:
            with st.status("🏗️ Building your market map...", expanded=True) as status:
                
                # 1. Attributes
                st.write("🔍 Determining competitive attributes...")
                if custom_attributes.strip():
                    attributes = [a.strip() for a in custom_attributes.split(",") if a.strip()]
                else:
                    attributes = generate_attributes(industry)
                
                # 2. Score
                st.write("📊 Scoring brands on perception...")
                feature_df = score_companies(all_companies, attributes, industry)
                
                # 3. Reduce
                st.write("🗺️ Calculating 2D coordinates...")
                coords_df = reduce_dimensions(feature_df, method=method)
                
                # 4. Insights
                st.write("💡 Generating strategic insights...")
                insights = generate_insights(coords_df, target_company, feature_df)
                
                status.update(label="Analysis Complete!", state="complete", expanded=False)

            # --- Layout: Map & Insights ---
            col1, col2 = st.columns([2, 1])

            with col1:
                st.header("1. Perceptual Map")
                
                # Add highlighting for the target company
                coords_df['Type'] = ['Target Brand' if i == target_company else 'Competitor' for i in coords_df.index]
                
                fig = px.scatter(
                    coords_df,
                    x="x", y="y",
                    text=coords_df.index,
                    color="Type",
                    color_discrete_map={'Target Brand': '#EF553B', 'Competitor': '#636EFA'},
                    title=f"Market Positioning: {industry}",
                    template="plotly_white",
                    height=600
                )
                fig.update_traces(textposition="top center", marker=dict(size=14, line=dict(width=2, color='DarkSlateGrey')))
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.header("2. Strategic Insights")
                st.markdown(insights)

            # --- Data Breakdown ---
            st.divider()
            with st.expander("View Detailed Attribute Scores (1-10)"):
                st.dataframe(feature_df.style.background_gradient(cmap='Blues', axis=1))

        except Exception as e:
            st.error(f"Something went wrong: {str(e)}")
