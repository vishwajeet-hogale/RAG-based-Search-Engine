import streamlit as st
from modules.module2.query_index import query_vector_db

st.set_page_config(page_title="Semantic Search Chat", layout="wide")
st.title("💬 Semantic Search Chat")
st.caption("Powered by SBERT + ChromaDB")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "searching" not in st.session_state:
    st.session_state.searching = False

# Input + Button (disabled while searching)
with st.form(key="query_form", clear_on_submit=True):
    user_input = st.text_input("Enter your question", placeholder="e.g. Applications of AI in healthcare", disabled=st.session_state.searching)
    submit_button = st.form_submit_button("Search", disabled=st.session_state.searching)

# When search button is clicked
if submit_button and user_input:
    st.session_state.searching = True
    st.session_state.chat_history.append(user_input)

    with st.spinner("🔍 Searching vector DB..."):
        results = query_vector_db(user_input, top_k=45)

    # Display results
    st.markdown("### 🔎 Results")
    for res in results:
        with st.container():
            st.markdown(
                f"""
                <div style="border:1px solid #CCC;padding:15px;border-radius:10px;margin-bottom:15px;background-color:#000000">
                    <h5 style="margin:0 0 10px;">📁 Source: {res['source']} &nbsp;&nbsp; 🆔 Doc ID: {res['doc_id']} &nbsp;&nbsp; ⭐ Score: {res['score']}</h5>
                    <p style="margin:0;font-size:16px;line-height:1.6;">{res['content']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.session_state.searching = False  # ✅ re-enable search

# Show past queries
if st.session_state.chat_history:
    st.sidebar.markdown("### 🕓 Past Queries")
    for q in reversed(st.session_state.chat_history[-5:]):
        st.sidebar.markdown(f"- {q}")
