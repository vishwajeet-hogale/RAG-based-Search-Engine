import streamlit as st
from modules.module2.roberta_query import search_expanded_query
from modules.module2.intent_classifier import detect_intent

st.set_page_config(page_title="Semantic Search Chat", layout="wide")
st.title("💬 Semantic Search Chat")
st.caption("Powered by Roberta + ChromaDB + Intent-Aware Ranking")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "searching" not in st.session_state:
    st.session_state.searching = False

# === Helper to sort results by intent-based preference ===
INTENT_PRIORITY = {
    "Research": ["Research", "Highlights", "Professors", "Labs", "Institutes"],
    "Professors": ["Professors", "Labs", "Research", "Highlights", "Institutes"],
    "Labs": ["Labs", "Professors", "Research", "Highlights", "Institutes"],
    "Institutes": ["Institutes", "Labs", "Professors", "Research", "Highlights"],
    "Research_current_highlights": ["Research_current_highlights", "Research", "Professors", "Labs", "Institutes"],
}
def rerank_by_intent(results, intent):
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    def score(item):
        doc, meta, dist = item
        source_priority = INTENT_PRIORITY.get(intent, [])
        base_score = 1 / (1 + dist)
        try:
            priority = source_priority.index(meta["source"])
        except ValueError:
            priority = len(source_priority)
        return -priority, base_score  # first by source type, then by semantic score

    ranked = sorted(zip(docs, metas, distances), key=score, reverse=True)
    return ranked

# Input + Button (disabled while searching)
with st.form(key="query_form", clear_on_submit=True):
    user_input = st.text_input("Enter your question", placeholder="e.g. Applications of AI in healthcare", disabled=st.session_state.searching)
    submit_button = st.form_submit_button("Search", disabled=st.session_state.searching)

# When search button is clicked
if submit_button and user_input:
    st.session_state.searching = True
    st.session_state.chat_history.append(user_input)

    with st.spinner("🔍 Expanding and searching..."):
        raw_results = search_expanded_query(user_input, top_k=45)
        intent = detect_intent(user_input)
        print(intent)
        ranked_results = rerank_by_intent(raw_results, intent)

    st.markdown(f"### 🔎 Top Results for Intent: `{intent}`")
    for i, (doc, meta, dist) in enumerate(ranked_results):
        score = round(1 / (1 + dist), 4)
        st.markdown(
            f"""
            <div style="border:1px solid #CCC;padding:15px;border-radius:10px;margin-bottom:15px;background-color:#000000">
                <h5 style="margin:0 0 10px;">📁 Source: {meta['source']} &nbsp;&nbsp; 🆔 Doc ID: {meta['doc_id']} &nbsp;&nbsp; ⭐ Score: {score}</h5>
                <p style="margin:0;font-size:16px;line-height:1.6;">{doc}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.session_state.searching = False

# Show past queries
if st.session_state.chat_history:
    st.sidebar.markdown("### 🕓 Past Queries")
    for q in reversed(st.session_state.chat_history[-5:]):
        st.sidebar.markdown(f"- {q}")
