import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from clustering_utils import GeneticAlgorithmClustering, load_iris_data, load_blobs_data, load_mall_data

# Page Config
st.set_page_config(
    page_title="GA Clustering Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Thematic" look
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    .metric-card {
        background-color: #0E1117;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #262730;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🧬 Clustering Config")

dataset_name = st.sidebar.radio("Select Dataset", ["Iris", "Synthetic Blobs", "Mall Customers"])

st.sidebar.markdown("### Genetic Algorithm Params")
n_pop = st.sidebar.slider("Population Size", 10, 200, 50, 10)
max_gen = st.sidebar.slider("Generations", 10, 500, 100, 10)
mut_rate = st.sidebar.slider("Mutation Rate", 0.0, 1.0, 0.1, 0.01)
sel_rate = st.sidebar.slider("Selection Rate", 0.1, 0.9, 0.5, 0.1)

# Main Content
st.title("🧬 Genetic Algorithm Clustering")
st.markdown(f"**Dataset:** {dataset_name} | **Objective:** Minimize Within-Cluster Sum of Squares (WCSS)")

# Load Data
X_scaled = None
scaler = None
actual_data_msg = ""
n_clusters_default = 3
feature_names = ["Feature 1", "Feature 2"]

if dataset_name == "Iris":
    X_scaled, scaler, _ = load_iris_data()
    n_clusters_default = 3
    feature_names = ["Sepal Length", "Sepal Width"] # Simplified for 2D plot, usually we plot first 2
    actual_data_msg = "Iris dataset detected (4 dimensions). plotting first 2 PCA dimensions or features."
    
elif dataset_name == "Synthetic Blobs":
    X_scaled, scaler, _ = load_blobs_data()
    n_clusters_default = 4
    feature_names = ["X", "Y"]

elif dataset_name == "Mall Customers":
    X_scaled, scaler, msg = load_mall_data()
    if X_scaled is None:
        st.error(msg)
    else:
        n_clusters_default = 5
        feature_names = ["Annual Income (k$)", "Spending Score (1-100)"]

if X_scaled is not None:
    k = st.sidebar.number_input("Number of Clusters (k)", 2, 10, n_clusters_default)
    
    if st.sidebar.button("🚀 Run Clustering", type="primary"):
        st.divider()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### ⚙️ Execution")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            ga = GeneticAlgorithmClustering(n_clusters=k, n_population=n_pop, max_generations=max_gen, mutation_rate=mut_rate, selection_rate=sel_rate)
            
            def on_gen(gen, total, fit):
                progress = gen / total
                progress_bar.progress(progress)
                status_text.text(f"Generation {gen}/{total} | Best Fitness: {fit:.2f}")
                # time.sleep(0.01) # Optional purely for visual effect

            start_time = time.time()
            best_centroids, history = ga.fit(X_scaled, progress_callback=on_gen)
            end_time = time.time()
            
            st.success(f"Converged in {end_time - start_time:.2f}s")
            
            # Metrics
            st.metric("Final WCSS (Fitness)", f"{ga.best_fitness:.2f}")

        with col2:
            st.markdown("### 📈 Convergence History")
            hist_df = pd.DataFrame({"Generation": range(len(history)), "Fitness": history})
            fig_hist = px.line(hist_df, x="Generation", y="Fitness", template="plotly_dark")
            fig_hist.update_layout(height=300)
            st.plotly_chart(fig_hist, use_container_width=True)

        st.divider()
        
        # Cluster Visualization
        st.subheader("🔍 Clustering Results")
        
        labels = ga.assign_clusters(X_scaled, best_centroids)
        
        # Prepare dataframe for plotting
        if dataset_name == "Mall Customers":
             # Inverse transform for real values
             X_real = scaler.inverse_transform(X_scaled)
             centroids_real = scaler.inverse_transform(best_centroids)
             plot_df = pd.DataFrame(X_real, columns=feature_names)
             
        elif dataset_name == "Iris":
            # Just take first 2 dims for vis or use PCA if we wanted to be fancy, 
            # but let's stick to simple slicing for speed as Utils didn't do PCA.
            # Actually, notebook plotted first 2 dims? Notebook used scatter(x=data[:,0], y=data[:,1])
             X_real = X_scaled # Keeping scaled for standard viz or inverse. Let's inverse.
             X_real = scaler.inverse_transform(X_scaled)
             # Iris has 4 cols. We construct DF with all.
             plot_df = pd.DataFrame(X_real, columns=["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"])
             centroids_real = scaler.inverse_transform(best_centroids)

        else:
             X_real = scaler.inverse_transform(X_scaled)
             plot_df = pd.DataFrame(X_real, columns=feature_names)
             centroids_real = scaler.inverse_transform(best_centroids)

        plot_df["Cluster"] = labels.astype(str)
        
        # 2D Scatter
        if dataset_name == "Iris":
             fig_scatter = px.scatter(plot_df, x="Sepal Length", y="Sepal Width", color="Cluster", 
                                      title="Iris: Sepal Length vs Width", template="plotly_dark",
                                      hover_data=["Petal Length", "Petal Width"])
        else:
             fig_scatter = px.scatter(plot_df, x=feature_names[0], y=feature_names[1], color="Cluster", 
                                      title=f"{dataset_name} Clusters", template="plotly_dark")
        
        # Add centroids
        # Plotly express doesn't easily add second trace without graph objects
        # We overlay centroids
        if dataset_name == "Iris":
            fig_scatter.add_trace(go.Scatter(x=centroids_real[:, 0], y=centroids_real[:, 1], mode='markers', 
                                             marker=dict(symbol='x', size=12, color='white', line=dict(width=2)),
                                             name='Centroids'))
        else:
             fig_scatter.add_trace(go.Scatter(x=centroids_real[:, 0], y=centroids_real[:, 1], mode='markers', 
                                             marker=dict(symbol='x', size=12, color='white', line=dict(width=2)),
                                             name='Centroids'))

        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Insights Expansion
        with st.expander("💡 Insights & Observations", expanded=True):
            if dataset_name == "Iris":
                st.markdown("""
                - **Observations:**
                    - Convergence typically shows a sharp initial decrease in WCSS.
                    - Overlap between Versicolor and Virginica makes perfect separation difficult.
                - **Performance:**
                    - GA successfully identifies the 3 groups roughly corresponding to species.
                """)
            elif dataset_name == "Synthetic Blobs":
                st.markdown("""
                - **Observations:**
                    - Distinct, well-separated clusters are easily found by GA.
                - **Performance:**
                    - Fast and stable convergence expected.
                """)
            elif dataset_name == "Mall Customers":
                st.markdown("""
                - **Observations:**
                    - Standard analysis suggests 5 clusters (Low/Low, Low/High, Mid/Mid, High/Low, High/High).
                    - GA finds these natural market segments.
                - **Business Value:**
                    - **Target Group:** High Income, High Spending Score.
                    - **Conservative:** High Income, Low Spending Score.
                """)
                
else:
    st.info("Please run the app with a valid dataset selected.")
