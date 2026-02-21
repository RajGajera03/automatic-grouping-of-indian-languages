import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import umap
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sentence_transformers import SentenceTransformer
import os

# Page Config
st.set_page_config(page_title="Indian Language Grouping", layout="wide")

st.title("🇮🇳 Indian Language Grouping with Embeddings")
st.markdown("""
This dashboard visualizes semantic relationships between Indian languages using **LaBSE** embeddings.
Data is clustered to see if it aligns with **Indo-Aryan** vs **Dravidian** language families.
""")

# --- Data Loading ---
@st.cache_data
def load_data():
    # Robust path resolution using script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Potential paths for embeddings.npy
    paths = [
        os.path.join(script_dir, "..", "data", "embeddings.npy"),  # Data is in root/data/
        os.path.join(script_dir, "data", "embeddings.npy"),        # Data is in same dir as script
        "data/embeddings.npy",                                     # Relative to current working dir
        "embeddings.npy"
    ]
    
    emb_path = None
    for p in paths:
        if os.path.exists(p):
            emb_path = p
            break
            
    if emb_path:
        csv_path = emb_path.replace("embeddings.npy", "metadata.csv")
        if os.path.exists(csv_path):
            try:
                embeddings = np.load(emb_path)
                df = pd.read_csv(csv_path)
                # Ensure text column exists for hover
                if "text" not in df.columns and "sentence" in df.columns:
                    df["text"] = df["sentence"] 
                return embeddings, df
            except Exception as e:
                st.error(f"Error loading data: {e}")
                return None, None
            
    return None, None

embeddings, df = load_data()

if embeddings is None:
    st.error("❌ Data not found! Please run the notebooks (1 & 2) to generate embeddings.")
    st.stop()

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Interactive Visualization", "⚠️ Misclassification Analysis", "🧪 Playground"])

# --- Tab 1: Visualization ---
with tab1:
    st.subheader("Language Clusters (UMAP)")
    st.write("Each point is a sentence. Colors represent languages, and shapes represent families.")
    
    # Run UMAP (Cached)
    @st.cache_data
    def get_umap_data(emb):
        # Sample if too large for speed, but UMAP is faster than t-SNE
        if len(emb) > 10000:
            indices = np.random.choice(len(emb), 10000, replace=False)
            emb_subset = emb[indices]
        else:
            indices = None
            emb_subset = emb
            
        reducer = umap.UMAP(
            n_components=2, 
            n_neighbors=15, 
            min_dist=0.1, 
            metric='cosine', 
            random_state=42
        )
        return reducer.fit_transform(emb_subset), indices

    with st.spinner("Running UMAP dimensionality reduction..."):
        embeddings_2d, indices = get_umap_data(embeddings)
    
    # Filter df if subsampled
    plot_df = df.iloc[indices].copy() if indices is not None else df.copy()
    
    plot_df['UMAP 1'] = embeddings_2d[:, 0]
    plot_df['UMAP 2'] = embeddings_2d[:, 1]
    
    fig = px.scatter(
        plot_df, 
        x='UMAP 1', 
        y='UMAP 2', 
        color='language', 
        symbol='Family', 
        title='Indian Languages Clustering (UMAP Projection)',
        hover_data=['text'],
        template='plotly_dark',
        height=700
    )
    fig.update_traces(marker=dict(size=5, opacity=0.7))
    st.plotly_chart(fig, use_container_width=True)

# --- Tab 2: Misclassification ---
with tab2:
    st.subheader("Misclassified Languages")
    st.write("Identifying languages that cluster differently from their known family.")
    
    k = st.slider("Number of Clusters (K)", 2, 5, 2)
    
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['cluster'] = kmeans.fit_predict(embeddings)
    
    # Analyze clusters
    cluster_map = {}
    st.write("### Cluster Composition")
    cols = st.columns(k)
    
    for c in range(k):
        cluster_data = df[df['cluster'] == c]
        dominant = cluster_data['Family'].mode()[0] if not cluster_data.empty else "Unknown"
        cluster_map[c] = dominant
        
        with cols[c]:
            st.metric(f"Cluster {c}", dominant)
            st.caption(f"{len(cluster_data)} sentences")
    
    # Find misclassifications (Only makes sense for K=2 usually, or if we map strictly)
    if k == 2:
        df['predicted_family'] = df['cluster'].map(cluster_map)
        misclassified = df[df['Family'] != df['predicted_family']]
        
        if not misclassified.empty:
            st.warning(f"Found {len(misclassified)} potential misclassifications.")
            st.dataframe(misclassified[['language', 'Family', 'predicted_family', 'text']], use_container_width=True)
        else:
            st.success("Clean separation! No misclassifications found.")
    else:
        st.info("Misclassification table is only available for K=2 (Indo-Aryan vs Dravidian).")

# --- Tab 3: Playground ---
with tab3:
    st.subheader("Test Your Own Sentence")
    
    # Load Model (Cached)
    @st.cache_resource
    def load_model():
        return SentenceTransformer("sentence-transformers/LaBSE")

    with st.spinner("Loading AI Model (LaBSE)..."):
        model = load_model()
    
    # Train simple classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(embeddings, df['Family'])
    
    user_text = st.text_area("Enter a sentence (Hindi, Tamil, Telugu, etc.):", "आप कैसे हैं?")
    
    if st.button("Analyze Sentence"):
        with st.spinner("Analyzing..."):
            user_emb = model.encode([user_text])
            
            # Predict
            pred_family = knn.predict(user_emb)[0]
            
            # Display
            st.markdown(f"### Predicted Family: **:blue[{pred_family}]**")
            
            # Find nearest neighbors
            distances, indices = knn.kneighbors(user_emb, n_neighbors=3)
            st.write("---")
            st.write("**Most Similar Sentences in Dataset:**")
            
            for dist, idx in zip(distances[0], indices[0]):
                row = df.iloc[idx]
                st.info(f"**{row['language']}** ({row['Family']}): {row['text']} \n\n *(score: {1/(1+dist):.2f})*")
