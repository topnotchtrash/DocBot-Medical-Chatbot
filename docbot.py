import streamlit as st

# Ensure this is the first Streamlit command
st.set_page_config(page_title="DocBot - Medical Chatbot", page_icon="🩺", layout="centered")

from LLM_Connect_Memory import create_qa_chain


# Cache chatbot initialization
@st.cache_resource
def load_qa_chain():
    return create_qa_chain()

qa_chain = load_qa_chain()

# Styled title
st.markdown("""
    <h1 style='text-align: center; color: #2c3e50;'>🩺 DocBot - Your Medical Assistant</h1>
    <p style='text-align: center; font-size: 18px;'>Ask medical-related questions, and I'll provide a clear and helpful response.</p>
""", unsafe_allow_html=True)

# Function to clean chatbot response
def format_chatbot_response(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\d+\.", "", text)  # Remove numbered lists
    text = re.sub(r"-", "", text)  # Remove dashes
    text = re.sub(r"\s+", " ", text).strip()  # Clean spaces
    return text

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for message in st.session_state.messages:
    role, content = message["role"], message["content"]
    if role == "user":
        st.markdown(f"👤 **You:** {content}")
    else:
        st.markdown(f"🤖 **DocBot:** {content}")

# User query input
user_query = st.chat_input("📝 Type your medical query here...")

if user_query:
    st.markdown(f"👤 **You:** {user_query}")
    with st.spinner("🔄 Fetching answer..."):
        response = qa_chain.invoke({"query": user_query})
    
    formatted_response = format_chatbot_response(response["result"])
    
    # Store conversation in session history
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.session_state.messages.append({"role": "bot", "content": formatted_response})
    
    # Display chatbot response
    st.markdown(f"🤖 **DocBot:** {formatted_response}")

# Footer with medical theme
st.markdown("""
    <hr>
    <p style='text-align: center; font-size: 16px;'>Made with ❤️ for better healthcare assistance</p>
""", unsafe_allow_html=True)