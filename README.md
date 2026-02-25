## Competitive-Landscape-Mapping

#### An intelligent competitive intelligence tool that leverages LLMs and dimensionality reduction to visualize market positioning. This app transforms raw brand names into a 2D strategic map, revealing where companies stand against their competitors.
---
### Project Concept
Traditional perceptual mapping requires expensive manual surveys. This tool automates the process by using **LLMs as a proxy for general market perception**, then applying data science techniques to plot the results.

### **The Workflow:**
1.  **User Input:** Enter your company, competitors, and industry.
2.  **LLM Synthesis:** The tool generates industry-specific attributes and scores each company (1–10).
3.  **Analytics Engine:** High-dimensional scores are compressed into 2D coordinates using **PCA** or **MDS**.
4.  **Visual Output:** A dynamic map showing strategic clusters and market "white space."
---
### Core Components

#### 1. Dimensionality Reduction
To turn 10+ attributes into a 2D map, the app utilizes:
*   **PCA (Principal Component Analysis):** Best for understanding which metrics drive market variance.
*   **MDS (Multidimensional Scaling):** Best for preserving the relative "distance" (similarity) between brands.

#### 2. Insights
The app doesn't just show a map; it interprets it. Using the resulting coordinates, the LLM provides summaries like:
*   *"Your brand is closest to Competitor A on perceived quality."*
*   *"There is an untapped gap in the 'Low Cost / High Innovation' quadrant."*

---

