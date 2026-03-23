"""
Streamlit dashboard for NLP sentiment analysis of product reviews.
Pages: Live prediction, Word clouds, Model comparison, Predictive terms, Data explorer.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="Sentiment analysis dashboard", layout="wide")

DATA_PATH = "data/product_reviews.csv"
OUTPUTS_DIR = "outputs"
MODELS_DIR = "models"
CLASS_NAMES = ["negative", "neutral", "positive"]


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


@st.cache_data
def load_model_comparison():
    path = os.path.join(OUTPUTS_DIR, "model_comparison.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return None


@st.cache_resource
def load_model_and_vectorizer():
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    tfidf_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
    if os.path.exists(model_path) and os.path.exists(tfidf_path):
        model = joblib.load(model_path)
        tfidf = joblib.load(tfidf_path)
        return model, tfidf
    return None, None


# --- Sidebar navigation ---
page = st.sidebar.radio(
    "Navigate",
    ["Live prediction", "Word clouds", "Model comparison", "Predictive terms", "Data explorer"],
)

df = load_data()
model, tfidf = load_model_and_vectorizer()


# =====================================================================
# PAGE 1: LIVE PREDICTION
# =====================================================================
if page == "Live prediction":
    st.title("Sentiment prediction")
    st.markdown("Enter a product review to classify its sentiment in real time.")

    user_text = st.text_area(
        "Type or paste a review below",
        value="This product is great, I love the quality and it works perfectly.",
        height=120,
    )

    if st.button("Predict sentiment") and user_text.strip():
        if model is not None and tfidf is not None:
            import re
            try:
                from nltk.corpus import stopwords
                from nltk.stem import WordNetLemmatizer
                stops = set(stopwords.words("english"))
            except LookupError:
                import nltk
                nltk.download("stopwords", quiet=True)
                nltk.download("wordnet", quiet=True)
                from nltk.corpus import stopwords
                from nltk.stem import WordNetLemmatizer
                stops = set(stopwords.words("english"))

            lemmatizer = WordNetLemmatizer()
            clean = user_text.lower()
            clean = re.sub(r"[^a-z\s]", "", clean)
            tokens = [lemmatizer.lemmatize(t) for t in clean.split() if t not in stops and len(t) > 2]
            clean_text = " ".join(tokens)

            X_vec = tfidf.transform([clean_text])

            # Get prediction
            pred_label = model.predict(X_vec)[0]
            pred_name = CLASS_NAMES[pred_label]

            # Get confidence scores
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_vec)[0]
            elif hasattr(model, "decision_function"):
                decisions = model.decision_function(X_vec)[0]
                # Convert to pseudo-probabilities via softmax
                exp_d = np.exp(decisions - np.max(decisions))
                proba = exp_d / exp_d.sum()
            else:
                proba = np.zeros(3)
                proba[pred_label] = 1.0

            # Display result
            color_map = {"positive": "#22c55e", "neutral": "#f59e0b", "negative": "#ef4444"}
            st.markdown(f"### Predicted sentiment: <span style='color:{color_map[pred_name]}'>{pred_name.upper()}</span>", unsafe_allow_html=True)

            # Confidence bar chart
            conf_df = pd.DataFrame({
                "Sentiment": CLASS_NAMES,
                "Confidence": proba,
            })
            fig = px.bar(
                conf_df, x="Sentiment", y="Confidence",
                color="Sentiment",
                color_discrete_map={"positive": "#22c55e", "neutral": "#f59e0b", "negative": "#ef4444"},
                title="Confidence scores",
            )
            fig.update_layout(yaxis_range=[0, 1], showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Train the model first. Run: `python -c \"from src.data_loader import load_and_prepare; from src.model import train_and_evaluate; X_tr, X_te, y_tr, y_te, _ = load_and_prepare(); train_and_evaluate(X_tr, X_te, y_tr, y_te)\"`")


# =====================================================================
# PAGE 2: WORD CLOUDS
# =====================================================================
elif page == "Word clouds":
    st.title("Word clouds by sentiment")
    st.markdown("Most frequent words for each sentiment class after text preprocessing.")

    try:
        from wordcloud import WordCloud
    except ImportError:
        st.error("Install wordcloud: `pip install wordcloud`")
        st.stop()

    import re
    try:
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        stops = set(stopwords.words("english"))
    except LookupError:
        import nltk
        nltk.download("stopwords", quiet=True)
        nltk.download("wordnet", quiet=True)
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        stops = set(stopwords.words("english"))

    lemmatizer = WordNetLemmatizer()

    def clean_text(text):
        text = text.lower()
        text = re.sub(r"[^a-z\s]", "", text)
        tokens = [lemmatizer.lemmatize(t) for t in text.split() if t not in stops and len(t) > 2]
        return " ".join(tokens)

    cols = st.columns(3)
    color_schemes = {
        "positive": "Greens",
        "neutral": "Oranges",
        "negative": "Reds",
    }

    for col, sent_class in zip(cols, CLASS_NAMES):
        with col:
            st.subheader(sent_class.capitalize())
            subset = df[df["sentiment"] == sent_class]
            all_text = " ".join(subset["review_text"].apply(clean_text).values)

            wc = WordCloud(
                width=400, height=300, background_color="white",
                colormap=color_schemes[sent_class],
                max_words=80, random_state=42,
            ).generate(all_text)

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)


# =====================================================================
# PAGE 3: MODEL COMPARISON
# =====================================================================
elif page == "Model comparison":
    st.title("Model comparison dashboard")

    comparison = load_model_comparison()
    if comparison is not None:
        st.subheader("Metrics overview")
        st.dataframe(
            comparison.style.highlight_max(axis=0, color="lightgreen"),
            use_container_width=True,
        )

        # Bar chart comparison
        fig = px.bar(
            comparison.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score"),
            x="index", y="Score", color="Metric",
            barmode="group",
            title="Model performance comparison",
            labels={"index": "Model"},
        )
        fig.update_layout(yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        # Show saved plots
        cm_path = os.path.join(OUTPUTS_DIR, "confusion_matrices.png")
        comp_path = os.path.join(OUTPUTS_DIR, "model_comparison.png")

        col1, col2 = st.columns(2)
        if os.path.exists(comp_path):
            col1.image(comp_path, caption="Model comparison", use_container_width=True)
        if os.path.exists(cm_path):
            col2.image(cm_path, caption="Confusion matrices", use_container_width=True)
    else:
        st.warning("Run the model pipeline first to generate results.")

    # Per-class metrics
    per_class_path = os.path.join(OUTPUTS_DIR, "per_class_metrics.csv")
    if os.path.exists(per_class_path):
        st.subheader("Per-class metrics")
        per_class = pd.read_csv(per_class_path)
        fig = px.bar(
            per_class, x="class", y="f1", color="model",
            barmode="group", title="F1 score by class and model",
        )
        st.plotly_chart(fig, use_container_width=True)


# =====================================================================
# PAGE 4: PREDICTIVE TERMS
# =====================================================================
elif page == "Predictive terms":
    st.title("Most predictive terms")
    st.markdown("Words with the highest TF-IDF coefficients for each sentiment class.")

    top_words_path = os.path.join(OUTPUTS_DIR, "top_words_per_class.csv")
    img_path = os.path.join(OUTPUTS_DIR, "top_predictive_words.png")

    if os.path.exists(img_path):
        st.image(img_path, caption="Top predictive words per class", use_container_width=True)

    if os.path.exists(top_words_path):
        top_words = pd.read_csv(top_words_path)

        selected_class = st.selectbox("Select sentiment class", CLASS_NAMES)
        class_words = top_words[top_words["class"] == selected_class].sort_values("coefficient", ascending=False)

        fig = px.bar(
            class_words, x="coefficient", y="word", orientation="h",
            color="coefficient",
            color_continuous_scale="YlOrRd" if selected_class == "negative" else "YlGn",
            title=f"Top predictive words for '{selected_class}' sentiment",
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Run the model pipeline to generate predictive term data.")


# =====================================================================
# PAGE 5: DATA EXPLORER
# =====================================================================
elif page == "Data explorer":
    st.title("Data explorer")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total reviews", f"{len(df):,}")
    col2.metric("Avg word count", f"{df['word_count'].mean():.0f}")
    col3.metric("Categories", f"{df['product_category'].nunique()}")
    col4.metric("Avg rating", f"{df['rating'].mean():.1f}")

    st.subheader("Sentiment distribution")
    sent_counts = df["sentiment"].value_counts()
    fig = px.pie(
        values=sent_counts.values, names=sent_counts.index,
        color=sent_counts.index,
        color_discrete_map={"positive": "#22c55e", "neutral": "#f59e0b", "negative": "#ef4444"},
        title="Sentiment class distribution",
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(
            df, x="word_count", color="sentiment", barmode="overlay", nbins=30,
            color_discrete_map={"positive": "#22c55e", "neutral": "#f59e0b", "negative": "#ef4444"},
            title="Word count distribution by sentiment",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.histogram(
            df, x="rating", color="sentiment", barmode="group",
            color_discrete_map={"positive": "#22c55e", "neutral": "#f59e0b", "negative": "#ef4444"},
            title="Rating distribution by sentiment",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Category breakdown")
    cat_sent = df.groupby(["product_category", "sentiment"]).size().reset_index(name="count")
    fig = px.bar(
        cat_sent, x="product_category", y="count", color="sentiment",
        barmode="group",
        color_discrete_map={"positive": "#22c55e", "neutral": "#f59e0b", "negative": "#ef4444"},
        title="Reviews by category and sentiment",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sample reviews")
    filter_sent = st.selectbox("Filter by sentiment", ["All"] + CLASS_NAMES)
    if filter_sent != "All":
        display_df = df[df["sentiment"] == filter_sent]
    else:
        display_df = df
    st.dataframe(display_df.head(20), use_container_width=True)
