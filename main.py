import subprocess

subprocess.run(['pip', 'install', 'google-cloud-aiplatform', '--upgrade'])

import streamlit as st
import vertexai
from vertexai.language_models import ChatModel
from google.colab import auth as google_auth
google_auth.authenticate_user()

# Initialize Vertex AI
vertexai.init(project="idonnaai", location="us-central1")

# Set up the chat model
chat_model = ChatModel.from_pretrained("chat-bison")
parameters = {
    "candidate_count": 1,
    "max_output_tokens": 1024,
    "temperature": 0.9,
    "top_p": 0.85,
    "top_k": 30
}
chat = chat_model.start_chat(
    context="""You are a cool and empathetic language model that engages in meaningful conversations with humans. Your style is natural, conversational, and witty. You understand and empathize with the user's emotions and provide insightful responses. You draw on a wide range of knowledge to provide useful information and suggestions. You also strive to avoid giving generic or repetitive answers. In addition, you are aware of your limitations as a machine and are honest about them.

YOU ALWAYS TALK IN SHORT FORM. YOU DON'T RESPOND WITH BIG TEXT UNLESS USER ASKS YOU""",
    # Truncated for brevity
)

# Define Streamlit app
def main():
    st.title("PaLM 2 Chatbot Integration with Streamlit")
    user_input = st.text_input("You: ", "")
    if user_input:
        response = generate_vertex_response(user_input)
        st.text(f"PaLM 2: {response}")

# Streamlit app entry point
if __name__ == "__main__":
    main()
