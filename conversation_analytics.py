import streamlit as st
import pandas as pd
import re
import docx2txt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import seaborn as sns
from matplotlib import rcParams
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import numpy as np
from sklearn.cluster import KMeans

st.set_page_config(page_title="Conversation Analysis Dashboard", page_icon="💬", layout="wide")

@st.cache_resource
def download_nltk_data():
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('vader_lexicon', quiet=True)

download_nltk_data()

@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    sia = SentimentIntensityAnalyzer()
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", framework="pt")
    semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
    return nlp, sia, summarizer, semantic_model

def clean_text(raw_text):
    raw_text = re.sub(r"(?i)participants:.*?(?=date:|time:|format:|[A-Z][a-z]+:)", "", raw_text, flags=re.S)
    raw_text = re.sub(r"(?i)date:.*|time:.*|format:.*", "", raw_text)
    raw_text = re.sub(r"\s+", " ", raw_text).strip()
    raw_text = raw_text.replace("\\'", "'").replace("'", "'")
    raw_text = re.sub(r"\b([A-Za-z]+'[A-Za-z]+)\s+\1\b", r"\1", raw_text, flags=re.IGNORECASE)
    raw_text = re.sub(r"\b(\w+)\s+\1\b", r"\1", raw_text, flags=re.IGNORECASE)
    raw_text = re.sub(r'\b(\w+\s+\w+)\s+\1\b', r'\1', raw_text, flags=re.IGNORECASE)
    filler_words = ["uh", "um", "erm", "hmm", "ah", "oh", "like", "you know", "i mean", "okay", "ok", "alright", "right", "yeah", "well", "so", "huh", "say", "feel", "think", "want", "really", "lot", "sure", "know", "okay", "just", "did", "said", "very", "good", "great", "nice", "right", "like", "yeah", "uh", "um", "thing"]
    pattern = r'\b(?:' + '|'.join(map(re.escape, filler_words)) + r')\b'
    raw_text = re.sub(pattern, "", raw_text, flags=re.IGNORECASE)
    raw_text = re.sub(r'\s+', ' ', raw_text).strip()
    return raw_text

def preprocess_text(text, nlp):
    extra_stopwords = {"say", "feel", "think", "want", "really", "lot", "sure", "know", "okay", "just", "did", "said", "very", "good", "great", "nice", "right", "like", "yeah", "uh", "um", "thing", "uh", "um", "erm", "hmm", "ah", "oh", "like", "you know", "i mean", "okay", "ok", "alright", "right", "yeah", "well", "so", "huh"}
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.pos_ in {"NOUN", "PROPN", "VERB", "ADJ"} and not token.is_stop and token.lemma_ not in extra_stopwords and len(token) > 2]
    return " ".join(tokens)

st.title("💬 Conversation Analysis Dashboard")
st.markdown("**Advanced NLP-powered conversation insights**")

uploaded_file = st.file_uploader("Upload a conversation file (.docx)", type=['docx'])

if uploaded_file is not None:
    with st.spinner("Loading models..."):
        nlp, sia, summarizer, semantic_model = load_models()
    
    with st.spinner("Processing conversation..."):
        text = docx2txt.process(uploaded_file)
        cleaned_text = clean_text(text)
        pattern = r"([A-Z][a-z]+):\s*(.*?)(?=[A-Z][a-z]+:|$)"
        matches = re.findall(pattern, cleaned_text)
        df = pd.DataFrame(matches, columns=["Speaker", "Dialogue"])
        df["clean_text"] = df["Dialogue"].apply(lambda x: preprocess_text(x, nlp))
        all_tokens = " ".join(df["clean_text"]).split()
        freqs = Counter(all_tokens)
        rare_tokens_df = []
        for text_ in df["clean_text"]:
            filtered = [w for w in text_.split() if freqs[w] < len(df) * 0.6]
            rare_tokens_df.append(" ".join(filtered))
        df["clean_text"] = rare_tokens_df
        df['text_length'] = df['clean_text'].apply(lambda x: len(str(x).split()))
        df["sentiment"] = df["clean_text"].apply(lambda x: sia.polarity_scores(str(x)))
        df["compound"] = df["sentiment"].apply(lambda x: x["compound"])
        df["positive"] = df["sentiment"].apply(lambda x: x["pos"])
        df["neutral"] = df["sentiment"].apply(lambda x: x["neu"])
        df["negative"] = df["sentiment"].apply(lambda x: x["neg"])
    
    st.success(f"✅ Processed {len(df)} dialogue entries from {df['Speaker'].nunique()} speakers")
    
    with st.expander("📄 View Conversation Data"):
        st.dataframe(df[["Speaker", "Dialogue", "compound"]].head(20))
    
    # create 7 tabs (keep your original first 6 names, then add Topic Dominance as tab7)
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Word Frequency",
        "☁️ Word Cloud",
        "🧠 Topic Modeling",
        "😊 Sentiment Analysis",
        "📈 Sentiment Flow",
        "🔗 Topic Continuity",
        "🔥 Topic Dominance"
    ])
    
    # --- Tab1: Word Frequency (unchanged) ---
    with tab1:
        st.header("Top Meaningful Words (TF-IDF)")
        vectorizer = TfidfVectorizer(max_features=30, stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(df["clean_text"])
        feature_names = vectorizer.get_feature_names_out()
        avg_tfidf = tfidf_matrix.mean(axis=0).A1
        tfidf_df = pd.DataFrame({"word": feature_names, "tfidf": avg_tfidf})
        tfidf_df = tfidf_df.sort_values("tfidf", ascending=False)
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.barh(tfidf_df.head(20)["word"], tfidf_df.head(20)["tfidf"], color="slateblue")
        ax1.invert_yaxis()
        ax1.set_title("Top Meaningful Words (Filtered by POS + TF-IDF + Frequency)")
        ax1.set_xlabel("TF-IDF Score")
        ax1.set_ylabel("Word / Phrase")
        st.pyplot(fig1)
    
    # --- Tab2: Word Cloud (unchanged) ---
    with tab2:
        st.header("Word Cloud of Meaningful Keywords")
        all_text = " ".join(df["clean_text"])
        wordcloud = WordCloud(width=1000, height=600, background_color="white", colormap="viridis", max_words=100, min_font_size=10).generate(all_text)
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        ax2.imshow(wordcloud, interpolation='bilinear')
        ax2.axis("off")
        ax2.set_title("Word Cloud of Meaningful Conversation Keywords", fontsize=16)
        st.pyplot(fig2)
    
    # --- Tab3: Topic Modeling (unchanged) ---
    with tab3:
        st.header("Topic Modeling (LDA)")
        vectorizer_lda = CountVectorizer(max_df=0.8, min_df=2, stop_words='english')
        X = vectorizer_lda.fit_transform(df["clean_text"])
        lda = LatentDirichletAllocation(n_components=3, random_state=42)
        lda.fit(X)
        words = vectorizer_lda.get_feature_names_out()
        for i, topic in enumerate(lda.components_):
            top_words = [words[j] for j in topic.argsort()[-8:]]
            st.write(f"**🪶 Topic {i+1}:** {', '.join(top_words)}")
    
    # --- Tab4: Sentiment Analysis by Speaker (unchanged) ---
    with tab4:
        st.header("Sentiment Analysis by Speaker")
        col1, col2 = st.columns(2)
        with col1:
            sentiment_summary = df.groupby("Speaker")[["compound", "positive", "neutral", "negative"]].mean().reset_index()
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            sns.barplot(data=sentiment_summary, x="Speaker", y="compound", palette="YlOrBr", ax=ax4)
            ax4.set_title("Average Sentiment (Compound Score) per Speaker")
            ax4.set_xlabel("Speaker")
            ax4.set_ylabel("Average Compound Score")
            ax4.axhline(0, color="gray", linestyle="--")
            plt.xticks(rotation=45)
            st.pyplot(fig4)
        with col2:
            text_count = df.groupby('Speaker')['text_length'].sum().sort_values(ascending=False).reset_index()
            fig4b, ax4b = plt.subplots(figsize=(10, 6))
            sns.barplot(data=text_count, x='Speaker', y='text_length', palette="YlOrBr", ax=ax4b)
            ax4b.set_title("Total Words Spoken by Each Speaker")
            ax4b.set_xlabel("Speaker")
            ax4b.set_ylabel("Word Count")
            plt.xticks(rotation=45)
            st.pyplot(fig4b)
    
    # --- Tab5: Sentiment Flow (unchanged) ---
    with tab5:
        st.header("Sentiment Flow Across Conversation")
        with st.spinner("Generating summaries..."):
            def summarize_with_speaker(row):
                text_ = row["clean_text"]
                speaker_ = row["Speaker"]
                try:
                    summary = summarizer(text_, max_length=20, min_length=8, do_sample=False, truncation=True)[0]['summary_text']
                    summary = summary[0].upper() + summary[1:]
                    return f"{speaker_} says: {summary}"
                except Exception:
                    return f"{speaker_} says: {text_[:80]}..."
            df["short_text"] = df.apply(summarize_with_speaker, axis=1)
        warm_palette = ["#FFD966", "#FFB347", "#FF8C42", "#E07B39", "#C65D21", "#8B4513"]
        fig5 = px.scatter(df, x=df.index, y="compound", color="Speaker", hover_data={"Speaker": True, "compound": True, "short_text": True}, title="Sentiment Flow Across Conversation (Interactive)", color_discrete_sequence=warm_palette)
        fig5.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.6)
        fig5.update_traces(mode="markers+lines", line=dict(width=3, shape='spline'), marker=dict(size=9, opacity=0.9, line=dict(width=1, color='black')))
        fig5.update_layout(xaxis_title="Dialogue Index", yaxis_title="Compound Sentiment Score", template="plotly_white", plot_bgcolor="rgba(255,255,240,0.9)")
        st.plotly_chart(fig5, use_container_width=True)
    
    # --- Tab6: Topic Continuity (user-provided code) ---
    with tab6:
        st.header("🧭 Topic Continuity (Context-Aware Semantic Similarity)")
        with st.spinner("Computing context-aware semantic similarity..."):
            # Load the exact model user specified
            model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

            # ---- Optional helper cleaning ----
            def clean_dialogue_text(text):
                text = text.lower()
                text = re.sub(r'[^a-z\s]', '', text)
                return text.strip()

            df["Dialogue_clean"] = df["Dialogue"].apply(clean_dialogue_text)

            # ---- Compute context-aware similarity ----
            similarities = []
            clarification_keywords = {"what", "why", "how", "where", "when", "huh", "sorry", "mean"}

            for i in range(len(df)):
                if i == 0:
                    similarities.append(None)
                    continue

                # Take the last 2–3 lines as conversational context
                context_window = " ".join(df["Dialogue_clean"].iloc[max(0, i-3):i].tolist())
                current = df["Dialogue_clean"].iloc[i]

                # Compute embeddings
                context_emb = model.encode(context_window, convert_to_tensor=True)
                current_emb = model.encode(current, convert_to_tensor=True)
                score = util.pytorch_cos_sim(current_emb, context_emb).item()

                # --- Soft adjustment for clarifications ---
                if any(word in current.split() for word in clarification_keywords):
                    score = min(1.0, score + 0.15)
                similarities.append(score)

            df["semantic_similarity"] = similarities

            # ---- Visualization prep ----
            df["short_text"] = df["Dialogue"].apply(
                lambda x: " ".join(x.split()[:15]) + "..." if len(x.split()) > 15 else x
            )

            # Base line plot
            fig6 = px.line(
                df,
                x=df.index,
                y="semantic_similarity",
                color="Speaker",
                hover_data={
                    "Speaker": True,
                    "semantic_similarity": True,
                    "short_text": True
                },
                title="🧭 Topic Continuity (Context-Aware Semantic Similarity)",
                color_discrete_sequence=["#F28E2B", "#E15759", "#76B7B2", "#59A14F", "#EDC948"]
            )

            # Add cross markers for low-similarity points
            low_points = df[df["semantic_similarity"] < 0.4]
            fig6.add_trace(
                go.Scatter(
                    x=low_points.index,
                    y=low_points["semantic_similarity"],
                    mode="markers",
                    marker=dict(
                        symbol="x",
                        color="red",
                        size=10,
                        line=dict(width=2),
                        opacity=0.7
                    ),
                    name="Digression (<0.4)",
                    hoverinfo="skip"  # skip hover so underlying line hover works
                )
            )

            # Add threshold and baseline
            fig6.add_hline(
                y=0.4,
                line_dash="dash",
                line_color="red",
                annotation_text="Digression Threshold (0.4)",
                annotation_position="top right"
            )
            fig6.add_hline(
                y=0.7,
                line_dash="dot",
                line_color="green",
                annotation_text="Strong Coherence (0.7)",
                annotation_position="bottom right"
            )

            # Styling
            fig6.update_traces(
                mode="markers+lines",
                line_shape="spline",
                line_width=3,
                marker=dict(size=8, opacity=0.9, line=dict(width=0.8, color="black")),
                hovertemplate="<b>%{customdata[0]}</b><br>Similarity: %{y:.2f}<br><i>%{customdata[1]}</i><extra></extra>"
            )

            fig6.update_layout(
                template="plotly_white",
                xaxis_title="Dialogue Index",
                yaxis_title="Semantic Similarity (Context-Aware)",
                hovermode="x unified",
                hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial"),
                plot_bgcolor="rgba(255,255,240,0.9)",
                paper_bgcolor="white",
                font=dict(family="Serif", size=14),
                legend=dict(
                    title="Speaker",
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="lightgray",
                    borderwidth=1
                )
            )

            st.plotly_chart(fig6, use_container_width=True)
    
    # --- Tab7: Topic Dominance (user-provided clustering & TF-IDF) ---
    with tab7:
        st.header("🔥 Topic Dominance in Conversation")
        with st.spinner("Computing topic clusters..."):
            model_dom = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
            embeddings = model_dom.encode(df["Dialogue"].tolist(), convert_to_numpy=True, show_progress_bar=True)

            # Step 2: number of clusters
            num_clusters = 5

            # Step 3: Cluster dialogues
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            df["topic_cluster"] = kmeans.fit_predict(embeddings)

            # Step 4: Name topics (auto label based on keywords)
            from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF_VEC
            def extract_top_words_per_cluster(texts, cluster_labels, n_words=5):
                vectorizer = TFIDF_VEC(stop_words="english", max_features=1000)
                X = vectorizer.fit_transform(texts)
                terms = np.array(vectorizer.get_feature_names_out())

                topic_keywords = []
                for cluster in np.unique(cluster_labels):
                    idx = np.where(cluster_labels == cluster)[0]
                    mean_tfidf = X[idx].mean(axis=0)
                    top_indices = np.argsort(mean_tfidf.A1)[::-1][:n_words]
                    top_words = ", ".join(terms[top_indices])
                    topic_keywords.append(top_words)
                return topic_keywords

            topic_labels = extract_top_words_per_cluster(df["Dialogue"], df["topic_cluster"])
            topic_map = {i: topic_labels[i] for i in range(num_clusters)}
            df["topic_label"] = df["topic_cluster"].map(topic_map)

            topic_counts = df["topic_label"].value_counts().reset_index()
            topic_counts.columns = ["Topic", "Count"]

            fig7 = px.bar(
                topic_counts,
                x="Count",
                y="Topic",
                orientation="h",
                color="Count",
                color_continuous_scale="YlOrBr",
                title="🔥 Dominant Topics in Conversation",
                text="Count"
            )

            fig7.update_layout(
                template="plotly_white",
                xaxis_title="Number of Dialogue Lines",
                yaxis_title="Extracted Topic",
                font=dict(family="Serif", size=14)
            )

            st.plotly_chart(fig7, use_container_width=True)

else:
    st.info("👆 Please upload a .docx conversation file to begin analysis")
    with st.expander("📋 Expected File Format"):
        st.markdown("""Your .docx file should contain conversations in this format:
        
Participants:
• Andy America (Manager): Senior Director of Product
• John Doe: QA Lead

Date: September 7, 2025
Time: 11:00 AM - 12:00 PM
Format: Video call

Andy: Hi John. Thanks for making the time.
Javier: Hello Andy. Of course. Everything is good on your end?

**Note:** Speaker names should be followed by a colon (:) and their dialogue.""")
