from agents.teaching_agent import teach
import streamlit as st
import os
from dotenv import load_dotenv
from pdf_processor import PDFProcessor
from embeddings import EmbeddingGenerator
from vector_store import PineconeVectorStore
from llm_client import GeminiLLMClient
from agents.query_understanding import QueryUnderstandingAgent
import uuid
from rapidfuzz import fuzz

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
    .section-header { font-size: 1.5rem; color: #2e8b57; margin-top: 2rem; margin-bottom: 1rem; }
    .info-box { background-color: #f0f8ff; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4; margin: 1rem 0; }
    .success-box { background-color: #f0fff0; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #32cd32; margin: 1rem 0; }
    .teaching-mode-explain { background-color: #e3f2fd; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #2196f3; margin: 1rem 0; }
    .teaching-mode-quiz { background-color: #fff3e0; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #ff9800; margin: 1rem 0; }
    .teaching-mode-summary { background-color: #e8f5e8; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #4caf50; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# Session state init
def initialize_session_state():
    if 'pdf_processed' not in st.session_state: st.session_state.pdf_processed = False
    if 'embeddings_uploaded' not in st.session_state: st.session_state.embeddings_uploaded = False
    if 'current_document_id' not in st.session_state: st.session_state.current_document_id = None
    if 'chat_history' not in st.session_state: st.session_state.chat_history = []
    if 'documents' not in st.session_state: st.session_state.documents = []
    # NEW: Add persistent teaching mode state
    if 'teaching_mode' not in st.session_state: st.session_state.teaching_mode = 'explain'
    if 'last_teaching_result' not in st.session_state: st.session_state.last_teaching_result = None
    if 'last_query' not in st.session_state: st.session_state.last_query = ""
    if 'show_results' not in st.session_state: st.session_state.show_results = False

# Check API keys
def check_api_keys():
    pinecone_key = os.getenv('PINECONE_API_KEY')
    pinecone_env = os.getenv('PINECONE_ENVIRONMENT')
    gemini_key = os.getenv('GEMINI_API_KEY')
    missing_keys = []
    if not pinecone_key: missing_keys.append("PINECONE_API_KEY")
    if not pinecone_env: missing_keys.append("PINECONE_ENVIRONMENT")
    if not gemini_key: missing_keys.append("GEMINI_API_KEY")
    return missing_keys, pinecone_key, pinecone_env, gemini_key

# Initialize components
def initialize_components(pinecone_key, pinecone_env, gemini_key):
    pdf_processor = PDFProcessor(chunk_size=1000, chunk_overlap=200)
    embedding_generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
    vector_store = PineconeVectorStore(pinecone_key, pinecone_env)
    llm_client = GeminiLLMClient(gemini_key)
    query_agent = QueryUnderstandingAgent(gemini_key, model_name="gemini-1.5-flash")

    connections_ok = True
    if not vector_store.initialize_pinecone(): connections_ok = False
    if not llm_client.initialize_gemini(): connections_ok = False

    if connections_ok:
        embedding_dim = embedding_generator.get_embedding_dimension()
        if not vector_store.create_index(embedding_dim): connections_ok = False

    return pdf_processor, embedding_generator, vector_store, llm_client, query_agent, connections_ok

# Display teaching response based on mode - ENHANCED VERSION
def display_teaching_response(teaching_result, query):
    """Display the teaching response with appropriate formatting - PERSISTENT VERSION"""
    
    if not teaching_result:
        st.warning("No teaching result to display")
        return "none"
    
    mode = teaching_result.get("mode", "explain")
    
    # Clear results button
    col_clear, col_spacer = st.columns([1, 4])
    with col_clear:
        if st.button("🗑️ Clear Results", key="clear_teaching_results"):
            st.session_state.last_teaching_result = None
            st.session_state.last_query = ""
            st.session_state.show_results = False
            st.rerun()
    
    # Display mode header with styling
    if mode == "explain":
        st.markdown('<div class="teaching-mode-explain">', unsafe_allow_html=True)
        st.markdown("### 🎯 Explanation Mode")
    elif mode == "quiz":
        st.markdown('<div class="teaching-mode-quiz">', unsafe_allow_html=True)
        st.markdown("### ❓ Quiz Mode")
    elif mode == "summary":
        st.markdown('<div class="teaching-mode-summary">', unsafe_allow_html=True)
        st.markdown("### 📝 Summary Mode")
    
    # Display content based on mode
    if mode == "quiz":
        quiz_questions = teaching_result.get("quiz", [])
        if quiz_questions and len(quiz_questions) > 0:
            st.success(f"✅ Generated {len(quiz_questions)} quiz questions!")
            st.markdown("**Quiz Questions:**")
            for i, q in enumerate(quiz_questions, 1):
                st.markdown(f"**Question {i}:** {q.get('question', 'No question available')}")
                
                # Display options if available
                options = q.get('options', [])
                if options:
                    for j, option in enumerate(options):
                        letter = chr(65 + j)  # A, B, C, D
                        st.markdown(f"  **{letter}.** {option}")
                
                # Show answer in expander
                with st.expander(f"💡 Show Answer for Question {i}"):
                    answer = q.get('answer', 'No answer provided')
                    st.markdown(f"**Correct Answer:** {answer}")
                    if 'explanation' in q and q['explanation']:
                        st.markdown(f"**Explanation:** {q['explanation']}")
                
                if i < len(quiz_questions):
                    st.markdown("---")
        else:
            st.warning("⚠️ No quiz questions were generated. The teaching agent may need debugging.")
            # DEBUG INFO for quiz
            st.error("DEBUG: Quiz generation failed")
            st.json(teaching_result)
    
    else:
        # For explain and summary modes, display the content
        content = teaching_result.get("content", "No content generated")
        if content and len(content.strip()) > 0:
            st.markdown(content)
        else:
            st.warning(f"No {mode} content was generated.")
            st.json(teaching_result)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return mode

# Main app
def main():
    initialize_session_state()

    st.markdown('<h1 class="main-header">🎓 AI Study Assistant - Agentic RAG</h1>', unsafe_allow_html=True)
    st.markdown("Upload PDF documents and ask questions about their content using AI-powered retrieval and generation.")

    missing_keys, pinecone_key, pinecone_env, gemini_key = check_api_keys()
    if missing_keys:
        st.error(f"Missing API keys: {', '.join(missing_keys)}")
        st.info("Please create a `.env` file with your API keys.")
        st.stop()

    with st.spinner("Initializing RAG components..."):
        pdf_processor, embedding_generator, vector_store, llm_client, query_agent, connections_ok = initialize_components(
            pinecone_key, pinecone_env, gemini_key
        )
    if not connections_ok:
        st.error("Failed to initialize one or more components. Please check your API keys and try again.")
        st.stop()

    with st.sidebar:
        st.markdown('<h2 class="section-header">📄 Document Upload</h2>', unsafe_allow_html=True)

        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to analyze"
        )

        # Process document button
        if uploaded_file and st.button("Process Document", type="primary"):
            try:
                chunks = pdf_processor.process_pdf(uploaded_file)
                if chunks:
                    embedded_chunks = embedding_generator.embed_chunks(chunks)
                    if embedded_chunks:
                        document_id = str(uuid.uuid4())
                        document_name = getattr(uploaded_file, 'name', 'Uploaded Document')
                        success = vector_store.upsert_embeddings(
                            embedded_chunks, 
                            document_id, 
                            document_name=document_name
                        )
                        if success:
                            # Add to session state documents list
                            st.session_state.documents.append({"id": document_id, "name": document_name})
                            st.session_state.pdf_processed = True
                            st.session_state.embeddings_uploaded = True
                            st.session_state.current_document_id = document_id
                            st.success(f"✅ Document '{document_name}' processed and uploaded successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to upload")
            except Exception as e:
                st.error(f"Error processing document: {e}")

        # Document selector
        if st.session_state.documents:
            st.markdown('<h2 class="section-header">📂 Documents</h2>', unsafe_allow_html=True)
            doc_labels = [f"{d['name']} ({d['id'][:8]}...)" for d in st.session_state.documents]
            current_idx = 0
            for i, d in enumerate(st.session_state.documents):
                if d['id'] == st.session_state.current_document_id:
                    current_idx = i
            selected_idx = st.selectbox(
                "Select document",
                list(range(len(doc_labels))),
                format_func=lambda i: doc_labels[i],
                index=current_idx
            )
            selected_doc = st.session_state.documents[selected_idx]
            st.session_state.current_document_id = selected_doc['id']

            if st.button("Delete Selected Document"):
                if vector_store.delete_document(selected_doc['id']):
                    st.session_state.documents = [
                        d for d in st.session_state.documents if d['id'] != selected_doc['id']
                    ]
                    if st.session_state.documents:
                        st.session_state.current_document_id = st.session_state.documents[0]['id']
                    else:
                        st.session_state.current_document_id = None
                        st.session_state.pdf_processed = False
                    st.success("Document deleted.")
                    st.rerun()

        # Settings
        st.markdown('<h2 class="section-header">⚙️ Settings</h2>', unsafe_allow_html=True)
        search_k = st.slider("Number of relevant chunks to retrieve", 1, 10, 5)
        
        # Teaching Mode Selection
        st.markdown('<h2 class="section-header">🎓 Teaching Preferences</h2>', unsafe_allow_html=True)
        default_mode = st.selectbox(
            "Default Teaching Mode",
            ["auto-detect", "explain", "quiz", "summary"],
            help="Choose default mode or let AI auto-detect from your question"
        )
        
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    # Main content
    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown('<h2 class="section-header">💬 Ask Questions</h2>', unsafe_allow_html=True)
        if not st.session_state.pdf_processed:
            st.info("Please upload and process a PDF document first.")
        else:
            # FIXED: Teaching mode selection with PERSISTENT radio buttons
            st.markdown("**🎯 Choose your learning style:**")
            
            teaching_mode = st.radio(
                "",
                ["explain", "quiz", "summary", "auto-detect"],
                horizontal=True,
                format_func=lambda x: {
                    'explain': '🎯 Detailed Explanation', 
                    'quiz': '❓ Quiz Questions',
                    'summary': '📝 Quick Summary',
                    'auto-detect': '🤖 Auto-Detect'
                }[x],
                key="teaching_mode_selector",
                index=["explain", "quiz", "summary", "auto-detect"].index(st.session_state.teaching_mode)
            )
            
            # Update session state when mode changes
            if teaching_mode != st.session_state.teaching_mode:
                st.session_state.teaching_mode = teaching_mode
            
            query = st.text_input(
                "Ask a question about your document:", 
                placeholder="What is the main topic of this document?", 
                key="query_input"
            )
            
            # FIXED: Process query with persistent results
            if st.button("🚀 Get Answer", type="primary") and query:
                if st.session_state.current_document_id:
                    status_box = st.status("Processing query...", state="running", expanded=True)
                    try:
                        status_box.update(label="Analyzing query…", state="running")
                        qa_analysis = query_agent.analyze_query(query)
                        with st.expander("🧭 Query understanding"):
                            st.write(qa_analysis)

                        questions_for_retrieval = qa_analysis.get("sub_questions") or [query]
                        status_box.update(label="Retrieving context…", state="running")

                        # Retrieve relevant chunks
                        aggregated, seen_ids = [], set()
                        for q in questions_for_retrieval:
                            q_embed = embedding_generator.generate_single_embedding(q)
                            if q_embed.size == 0:
                                continue
                            results = vector_store.search_similar(
                                q_embed, top_k=search_k,
                                document_id=st.session_state.current_document_id
                            )
                            for r in results:
                                if r.get('id') not in seen_ids:
                                    seen_ids.add(r.get('id'))
                                    r['matched_query'] = q
                                    aggregated.append(r)

                        if not aggregated:
                            status_box.update(label="No relevant context found.", state="error")
                            st.warning("Could not find relevant passages for your question.")
                            top_context = []
                        else:
                            # Rerank by combining vector and lexical score
                            reranked = []
                            for ch in aggregated:
                                vec_score = ch.get('score', 0.0)
                                base_q = ch.get('matched_query') or query
                                lex_score = fuzz.token_set_ratio(base_q, ch.get('text', '')) / 100.0
                                final_score = 0.7 * vec_score + 0.3 * lex_score
                                ch.update({
                                    "final_score": final_score,
                                    "lex_score": lex_score,
                                    "base_query": base_q
                                })
                                reranked.append(ch)
                            reranked.sort(key=lambda x: x['final_score'], reverse=True)
                            top_context = reranked[:search_k]

                            with st.expander("🔎 Retrieval diagnostics (top matches)"):
                                for idx, it in enumerate(top_context, start=1):
                                    st.markdown(
                                        f"**{idx}. Chunk {it.get('chunk_number', '?')}** — "
                                        f"vec: {it.get('score',0.0):.3f}, lex: {it.get('lex_score',0.0):.3f}, final: {it.get('final_score',0.0):.3f}\n"
                                        f"Matched query: {it.get('base_query', query)}"
                                    )
                                    st.write(
                                        (it.get('text','')[:300] + '...') 
                                        if len(it.get('text','')) > 300 else it.get('text','')
                                    )

                        # TEACHING MODE AGENT INTEGRATION - ENHANCED
                        status_box.update(label="Planning teaching approach…", state="running")

                        # Prepare structured content for teaching agent
                        structured_content = {
                            "topic": query,
                            "intent": qa_analysis.get("intent", "explain"),
                            "context": " ".join([chunk.get('text', '') for chunk in top_context[:3]]),
                            "relevant_chunks": top_context,
                            "sub_questions": qa_analysis.get("sub_questions", []),
                            "complexity": "intermediate",
                            "definition": "",
                            "steps": [],
                            "examples": [],
                            "summary": " ".join([chunk.get('text', '') for chunk in top_context[:2]])[:500]
                        }

                        # Determine teaching mode - USE RADIO BUTTON SELECTION
                        final_teaching_mode = teaching_mode
                        if final_teaching_mode == "auto-detect":
                            # Auto-detect from query analysis
                            detected_intent = qa_analysis.get("intent", "explain").lower()
                            final_teaching_mode = detected_intent if detected_intent in ["explain", "quiz", "summary"] else "explain"
                        
                        status_box.update(label=f"Generating {final_teaching_mode} content…", state="running")
                        st.info(f"🎯 Using {final_teaching_mode.title()} mode")

                        # Use Teaching Agent with enhanced error handling
                        try:
                            teaching_result = teach(
                                structured=structured_content,
                                mode=final_teaching_mode,
                                options={
                                    "num_questions": 3,  # Reduced for faster generation
                                    "summary_lines": 5   # for summary mode
                                }
                            )
                            
                            # Store in session state for persistence
                            st.session_state.last_teaching_result = teaching_result
                            st.session_state.last_query = query
                            st.session_state.show_results = True
                            
                        except Exception as teach_error:
                            st.error(f"Teaching agent error: {str(teach_error)}")
                            # Create fallback result
                            teaching_result = {
                                "mode": final_teaching_mode,
                                "content": f"Error in teaching agent. Here's the raw context: {structured_content['context'][:500]}...",
                                "quiz": []
                            }
                            st.session_state.last_teaching_result = teaching_result

                        status_box.update(label="Done ✅", state="complete")

                        # Update chat history
                        st.session_state.chat_history.append({
                            "query": query,
                            "answer": teaching_result.get("content", "") if teaching_result.get("mode") != "quiz" else f"Quiz with {len(teaching_result.get('quiz', []))} questions generated",
                            "teaching_mode": teaching_result.get("mode"),
                            "sources": [{"chunk_number": ch.get('chunk_number', 0), 
                                        "score": ch.get('final_score', 0), 
                                        "preview": ch.get('text', '')[:100] + "..."} 
                                       for ch in top_context[:3]]
                        })
                        
                        st.rerun()
                        
                    except Exception as e:
                        status_box.update(label=f"Error: {e}", state="error")
                        st.error(f"An error occurred: {str(e)}")

            # PERSISTENT RESULTS DISPLAY
            if st.session_state.show_results and st.session_state.last_teaching_result:
                st.markdown("---")
                st.markdown(f"### 📋 Results for: *{st.session_state.last_query}*")
                
                display_mode = display_teaching_response(st.session_state.last_teaching_result, st.session_state.last_query)
                
                # Follow-up options - ENHANCED
                if st.session_state.last_teaching_result and 'context' in locals():
                    st.markdown("### 💡 Try Different Learning Modes:")
                    col_alt1, col_alt2, col_alt3 = st.columns(3)

                    with col_alt1:
                        if st.button("🎯 Get Explanation", disabled=(display_mode == "explain")):
                            try:
                                explain_result = teach(structured_content, mode="explain")
                                st.session_state.last_teaching_result = explain_result
                                st.rerun()
                            except Exception as e:
                                st.error(f"Explanation error: {e}")

                    with col_alt2:
                        if st.button("❓ Take Quiz", disabled=(display_mode == "quiz")):
                            try:
                                quiz_result = teach(structured_content, mode="quiz", options={"num_questions": 3})
                                st.session_state.last_teaching_result = quiz_result
                                st.rerun()
                            except Exception as e:
                                st.error(f"Quiz error: {e}")

                    with col_alt3:
                        if st.button("📝 Get Summary", disabled=(display_mode == "summary")):
                            try:
                                summary_result = teach(structured_content, mode="summary")
                                st.session_state.last_teaching_result = summary_result
                                st.rerun()
                            except Exception as e:
                                st.error(f"Summary error: {e}")

        # Conversation history
        if st.session_state.chat_history:
            st.markdown('<h3 class="section-header">💭 Learning History</h3>', unsafe_allow_html=True)
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                teaching_mode = chat.get('teaching_mode', 'explain')
                mode_icon = {"explain": "🎯", "quiz": "❓", "summary": "📝"}.get(teaching_mode, "💬")
                
                with st.expander(f"{mode_icon} {teaching_mode.title()}: {chat['query'][:80]}...", expanded=(i==0)):
                    st.markdown(f"**Question:** {chat['query']}")
                    st.markdown(f"**Teaching Mode:** {teaching_mode.title()}")
                    st.markdown(f"**Response:** {chat['answer']}")
                    if chat.get('sources'):
                        st.markdown("**Sources:**")
                        for j, source in enumerate(chat['sources'][:3]):
                            st.markdown(f"- Chunk {source['chunk_number']} (Score: {source['score']:.3f})")
                            st.markdown(f"  *{source['preview']}*")

    with col2:
        st.markdown('<h2 class="section-header">📊 Document Info</h2>', unsafe_allow_html=True)
        if st.session_state.pdf_processed:
            stats = vector_store.get_index_stats()
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("**Document Status:** ✅ Processed")
            if stats: 
                st.markdown(f"**Total Chunks:** {stats.get('total_vector_count','Unknown')}")
            st.markdown(f"**Document ID:** `{st.session_state.current_document_id[:8]}...`")
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("**Embedding Model:** all-MiniLM-L6-v2")
            st.markdown("**LLM Model:** Gemini 1.5 Flash")
            st.markdown("**Vector Database:** Pinecone")
            st.markdown("**Teaching Agent:** ✅ Active")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No document processed yet.")

        # DEBUG PANEL - Remove after fixing
        if st.checkbox("🔧 Debug Teaching Agent", key="debug_teaching"):
            st.json({
                "current_mode": st.session_state.teaching_mode,
                "has_result": st.session_state.last_teaching_result is not None,
                "show_results": st.session_state.show_results,
                "last_query": st.session_state.last_query,
                "result_preview": str(st.session_state.last_teaching_result)[:200] if st.session_state.last_teaching_result else None
            })

        st.markdown('<h2 class="section-header">🎓 Learning Modes</h2>', unsafe_allow_html=True)
        st.markdown("""
        **🎯 Explain Mode:** Detailed step-by-step explanations with examples
        
        **❓ Quiz Mode:** Interactive multiple-choice questions to test understanding
        
        **📝 Summary Mode:** Concise bullet-point summaries of key concepts
        
        **🤖 Auto Mode:** Let AI choose the best teaching approach
        """)

        st.markdown('<h2 class="section-header">❓ How to Use</h2>', unsafe_allow_html=True)
        st.markdown("""
        1. **Upload PDF** and click "Process Document"
        2. **Choose Learning Style** with radio buttons
        3. **Ask Questions** about your document
        4. **Get Educational Responses** that persist until cleared
        5. **Switch Modes** using alternative mode buttons
        """)

if __name__ == "__main__":
    main()