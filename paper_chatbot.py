import streamlit as st
import json
from openai import OpenAI
from dotenv import load_dotenv
import os
import logging
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def load_paper_content():
    """Load the paper content from the JSON file."""
    try:
        with open("extracted_content/all_content.json", "r") as f:
            content = json.load(f)
            if not isinstance(content, dict):
                raise ValueError("Paper content must be a JSON object")
            return content
    except Exception as e:
        logger.error(f"Error loading paper content: {str(e)}")
        return None

def load_figure_analysis():
    """Load the figure analysis from the text file."""
    try:
        with open("figure_analysis/figure_page8_1_comprehensive_analysis.txt", "r") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error loading figure analysis: {str(e)}")
        return None

def chunk_text(text, chunk_size=1000):
    """
    Split text into chunks of approximately chunk_size words.
    Returns list of text chunks.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def get_relevant_chunks(query, chunks, top_k=3):
    """
    Get the most relevant text chunks for a given query using TF-IDF.
    """
    try:
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        
        # Fit and transform the chunks
        tfidf_matrix = vectorizer.fit_transform(chunks)
        
        # Transform the query
        query_vector = vectorizer.transform([query])
        
        # Calculate similarity
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
        # Get indices of top-k similar chunks
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Return the top-k chunks
        return [chunks[i] for i in top_indices]
    except Exception as e:
        logger.error(f"Error getting relevant chunks: {str(e)}")
        return chunks[:2]  # Fallback to first two chunks

def get_chatbot_response(messages, paper_content, figure_analysis):
    """Get response from the chatbot using OpenAI API with retrieval."""
    try:
        # Extract full text
        full_text = paper_content.get('full_text', '')
        
        # Chunk the text
        text_chunks = chunk_text(full_text)
        
        # Get the user's last message
        last_message = messages[-1]["content"] if messages else ""
        
        # Get relevant chunks
        relevant_chunks = get_relevant_chunks(last_message, text_chunks)
        
        # Join relevant chunks
        relevant_text = "\n\n".join(relevant_chunks)
        
        # Get figure captions (much smaller, can include all)
        captions = []
        for cap in paper_content.get("captions", []):
            if isinstance(cap, dict):
                captions.append(f"Figure {cap.get('figure_number', 'N/A')}: {cap.get('text', '')}")
        
        # Prepare the system message with the relevant text
        system_message = {
            "role": "system",
            "content": f"""You are a scientific paper analysis assistant. You have access to the following information:

            RELEVANT PAPER CONTENT:
            {relevant_text}

            FIGURE CAPTIONS:
            {'. '.join(captions)}

            FIGURE 8 ANALYSIS:
            {figure_analysis[:1000]}

            Your role is to:
            1. Help users understand the paper's findings
            2. Guide them through the methodology
            3. Explain the significance of the results
            4. Connect different parts of the paper
            5. Provide specific evidence from the text
            6. Help users formulate better questions about the paper

            Always:
            - Base your answers on the paper content
            - Cite specific sections or figures when relevant
            - Guide users toward deeper understanding
            - Suggest follow-up questions
            - Be precise and scientific in your language
            
            If you don't know the answer or it's not in the provided content, say so honestly.
            """
        }

        # Add system message to the conversation
        full_messages = [system_message] + messages
        
        logger.info(f"Token count estimate: {len(system_message['content'].split()) + sum(len(m.get('content', '').split()) for m in messages)}")

        # Get response from OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using smaller model to reduce token usage
            messages=full_messages,
            max_tokens=800,
            temperature=0.7
        )

        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"Error getting chatbot response: {str(e)}")
        return "I apologize, but I encountered an error while processing your request. Please try again."

def main():
    st.set_page_config(
        page_title="Paper Analysis Chatbot",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 Scientific Paper Analysis Chatbot")
    st.markdown("""
    Welcome to the Paper Analysis Chatbot! I can help you understand:
    - Key findings and conclusions
    - Methodology and techniques
    - Figure interpretations
    - Relationships between different parts of the paper
    - Specific evidence and data points
    
    Feel free to ask questions about any aspect of the paper!
    """)

    # Load paper content and figure analysis
    paper_content = load_paper_content()
    figure_analysis = load_figure_analysis()

    if not paper_content:
        st.error("Error loading paper content. Please check the file.")
        return

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the paper..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get chatbot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_chatbot_response(
                    st.session_state.messages,
                    paper_content,
                    figure_analysis if figure_analysis else ""
                )
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

    # Sidebar with paper information
    with st.sidebar:
        st.header("Paper Information")
        st.markdown("""
        ### Available Content:
        - Full paper text
        - Figure analysis
        - Methodology details
        - Key findings
        
        ### Suggested Questions:
        1. What are the main findings of this paper?
        2. Can you explain the methodology used?
        3. What do the figures tell us about the results?
        4. How do the findings relate to previous research?
        5. What are the limitations of this study?
        """)

if __name__ == "__main__":
    main() 