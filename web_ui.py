import streamlit as st
import requests
import os

# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(page_title="Hologram Knowledge Base", layout="wide")

st.title("🧠 Hologram Knowledge Base")

# Sidebar for KB Management
with st.sidebar:
    st.header("Knowledge Bases")
    
    # Upload new KB
    uploaded_file = st.file_uploader("Upload New KB (Text File)", type=["txt"])
    if uploaded_file is not None:
        if st.button("Upload"):
            files = {"file": (uploaded_file.name, uploaded_file, "text/plain")}
            try:
                response = requests.post(f"{API_URL}/kbs/upload", files=files)
                if response.status_code == 200:
                    st.success(f"Uploaded {uploaded_file.name}")
                    # Rerun to update list
                    st.rerun()
                else:
                    st.error("Upload failed")
            except Exception as e:
                st.error(f"Connection error: {e}")

    st.divider()

    # List and Select KB
    try:
        response = requests.get(f"{API_URL}/kbs")
        if response.status_code == 200:
            kbs = response.json().get("kbs", [])
        else:
            kbs = []
    except:
        kbs = []
        st.error("Could not connect to API Server")

    selected_kb = st.radio("Select Active KB", ["None"] + kbs)
    
    # Delete KB
    if selected_kb != "None":
        if st.button(f"Delete {selected_kb}"):
            try:
                response = requests.delete(f"{API_URL}/kbs/{selected_kb}")
                if response.status_code == 200:
                    st.success(f"Deleted {selected_kb}")
                    st.rerun()
                else:
                    st.error("Delete failed")
            except Exception as e:
                st.error(f"Error: {e}")
    
    # Load KB button (explicitly load the KB)
    if selected_kb != "None":
        if st.button("🔄 Load KB"):
            try:
                payload = {
                    "message": "load",
                    "kb_name": selected_kb
                }
                response = requests.post(f"{API_URL}/chat", json=payload)
                if response.status_code == 200:
                    st.success(f"Loaded {selected_kb}")
                else:
                    st.error("Load failed")
            except Exception as e:
                st.error(f"Error: {e}")

# Create tabs for different modes
tab1, tab2 = st.tabs(["💬 Chat", "🔍 Semantic Search"])

# TAB 1: Chat Interface
with tab1:
    st.header(f"Chatting with: {selected_kb}")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask a question..."):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Call API
        try:
            payload = {
                "message": prompt,
                "kb_name": selected_kb if selected_kb != "None" else None
            }
            response = requests.post(f"{API_URL}/chat", json=payload)
            
            if response.status_code == 200:
                reply = response.json().get("reply", "No response")
            else:
                reply = f"Error: {response.status_code}"
                
        except Exception as e:
            reply = f"Connection error: {e}"

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(reply)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": reply})

# TAB 2: Semantic Search
with tab2:
    st.header("🔍 Semantic Search")
    st.write("Search for concepts semantically related to your keywords")
    
    # Search input
    col1, col2 = st.columns([4, 1])
    with col1:
        search_query = st.text_input("Enter keywords", placeholder="e.g., speed of light")
    with col2:
        top_k = st.number_input("Results", min_value=1, max_value=20, value=10)
    
    if st.button("🔍 Search", type="primary"):
        if not search_query:
            st.warning("Please enter a search query")
        else:
            try:
                payload = {
                    "query": search_query,
                    "top_k": top_k
                }
                response = requests.post(f"{API_URL}/search", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                    
                    if results:
                        st.success(f"Found {len(results)} semantic matches")
                        
                        # Display results
                        for i, result in enumerate(results, 1):
                            score_percentage = result["score"] * 100
                            
                            # Color code based on similarity
                            if score_percentage >= 80:
                                color = "🟢"
                            elif score_percentage >= 60:
                                color = "🟡"
                            else:
                                color = "🔵"
                            
                            with st.container():
                                col_content, col_score = st.columns([4, 1])
                                with col_content:
                                    st.markdown(f"**{i}.** {result['content']}")
                                with col_score:
                                    st.markdown(f"{color} **{score_percentage:.1f}%**")
                                
                                # Show relations if available
                                relations = result.get('relations')
                                if relations:
                                    with st.expander(f"🔗 {len(relations)} related concepts"):
                                        for rel in relations:
                                            rel_strength = rel['strength'] * 100
                                            st.markdown(f"- **{rel['concept']}** (strength: {rel_strength:.1f}%)")
                                
                                st.divider()
                    else:
                        st.info("No results found")
                        
                elif response.status_code == 400:
                    error = response.json().get("detail", "Unknown error")
                    st.error(f"❌ {error}")
                    st.info("💡 Make sure to load a KB first using the 'Load KB' button in the sidebar")
                else:
                    st.error(f"Error: {response.status_code}")
                    
            except Exception as e:
                st.error(f"Connection error: {e}")
                st.info("Make sure the API server is running on http://localhost:8000")

