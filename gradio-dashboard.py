"""
gradio-dashboard.py
====================
Gradio dashboard for the Semantic Book Recommender.

Run with:
    python gradio-dashboard.py

Requires:
    - data/books_with_emotions.csv  (output of notebook 4)
    - data/tagged_description.txt   (output of notebook 2)
    - data/chroma_db/               (built in notebook 2)
    - .env containing GOOGLE_API_KEY
"""

import os
import pandas as pd
import gradio as gr
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

DATA_PATH        = "data/books_with_emotions.csv"
CHROMA_PATH      = "data/chroma_db"
PLACEHOLDER_IMG  = "https://placehold.co/128x192?text=No+Cover"
TOP_K_CANDIDATES = 50   # number of vector-search results before filtering
TOP_N_RESULTS    = 16   # final number of books shown in the gallery

# Emotion columns must match the names saved in notebook 4
TONE_MAP = {
    "Happy":       "joy",
    "Suspenseful": "fear",
    "Sad":         "sadness",
    "Surprising":  "surprise",
    "Angry":       "anger",
}

# ---------------------------------------------------------------------------
# Load data & vector store (once at startup)
# ---------------------------------------------------------------------------

print("Loading enriched book dataset...")
df = pd.read_csv(DATA_PATH)
print(f"  {len(df)} books loaded.")

print("Connecting to Chroma vector store with Gemini embeddings...")
# At query time we use task_type="retrieval_query" so Gemini optimises
# the query embedding to match against the indexed documents
query_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    task_type="retrieval_query",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)
db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=query_embeddings,
)
print("  Vector store ready.")


# ---------------------------------------------------------------------------
# Core recommendation logic
# ---------------------------------------------------------------------------

def retrieve_semantic_recommendations(
    query: str,
    category: str,
    tone: str,
) -> pd.DataFrame:
    """
    1. Semantic search: fetch top-k candidate books from ChromaDB.
    2. Parse ISBNs from the raw document text (the 'tag_description' hack).
    3. Filter by user-selected category.
    4. Sort by user-selected emotional tone.
    5. Return the top 16 results.
    """

    # --- Step 1: semantic search ---
    results = db.similarity_search(query, k=TOP_K_CANDIDATES)

    # --- Step 2: extract ISBNs ---
    # Each document's page_content = "<isbn13> <description text>"
    # The ISBN is always the very first space-delimited token.
    isbn_list = []
    for doc in results:
        first_token = doc.page_content.split(" ")[0]
        try:
            isbn_list.append(int(first_token))
        except ValueError:
            continue

    recommendations = df[df["isbn13"].isin(isbn_list)].copy()

    # --- Step 3: category filter ---
    if category != "All":
        category_map = {"Fiction": "fiction", "Non-Fiction": "non-fiction"}
        target = category_map.get(category)
        if target:
            recommendations = recommendations[
                recommendations["simple_category"] == target
            ]

    # --- Step 4: tone sort ---
    if tone != "All" and tone in TONE_MAP:
        sort_col = TONE_MAP[tone]
        recommendations = recommendations.sort_values(by=sort_col, ascending=False)

    return recommendations.head(TOP_N_RESULTS)


# ---------------------------------------------------------------------------
# Gradio wrapper
# ---------------------------------------------------------------------------

def truncate(text: str, max_words: int = 30) -> str:
    """Truncate text to max_words words with trailing ellipsis."""
    words = str(text).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + "..."


def on_submit(query: str, category: str, tone: str):
    """Called on button click. Returns a list of (image_url, caption) tuples."""
    if not query.strip():
        return []

    recs = retrieve_semantic_recommendations(query, category, tone)

    gallery_items = []
    for _, row in recs.iterrows():
        # Thumbnail with fallback
        thumbnail = str(row.get("thumbnail", "")).strip()
        if not thumbnail or thumbnail == "nan":
            thumbnail = PLACEHOLDER_IMG

        # Caption: title + author + truncated description
        short_desc = truncate(row.get("description", ""), max_words=30)
        authors    = row.get("authors", "Unknown Author")
        caption    = f"{row['title']}\nby {authors}\n\n{short_desc}"

        gallery_items.append((thumbnail, caption))

    return gallery_items


# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------

with gr.Blocks(theme=gr.themes.Glass()) as demo:

    gr.Markdown("## 📚 Semantic Book Recommender")
    gr.Markdown(
        "Describe the kind of book you're looking for in plain English. "
        "Optionally narrow results by category or emotional tone."
    )

    with gr.Row():

        # ── Left panel: inputs ──────────────────────────────────────────────
        with gr.Column(scale=1, min_width=260):

            query_input = gr.Textbox(
                label="What kind of book are you looking for?",
                placeholder="e.g. A story about forgiveness and second chances",
                lines=3,
            )
            category_dropdown = gr.Dropdown(
                label="Category",
                choices=["All", "Fiction", "Non-Fiction"],
                value="All",
            )
            tone_dropdown = gr.Dropdown(
                label="Emotional Tone",
                choices=["All", "Happy", "Suspenseful", "Sad", "Surprising", "Angry"],
                value="All",
            )
            submit_btn = gr.Button("Find Books", variant="primary")

        # ── Right panel: results gallery ────────────────────────────────────
        with gr.Column(scale=3):
            gallery_output = gr.Gallery(
                label="Recommended Books",
                columns=4,
                rows=4,
                object_fit="cover",
                height="auto",
                show_label=False,
            )

    submit_btn.click(
        fn=on_submit,
        inputs=[query_input, category_dropdown, tone_dropdown],
        outputs=gallery_output,
    )

    # Allow pressing Enter in the query box to trigger the search too
    query_input.submit(
        fn=on_submit,
        inputs=[query_input, category_dropdown, tone_dropdown],
        outputs=gallery_output,
    )

if __name__ == "__main__":
    demo.launch()
