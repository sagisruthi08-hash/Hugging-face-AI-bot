```python
import streamlit as st
from transformers import pipeline
import torch

st.set_page_config(
    page_title="Mentora AI",
    page_icon="🤖",
    layout="centered"
)

st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: gray;
        margin-bottom: 30px;
    }

    .stChatMessage {
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">🤖 Mentora AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Your Intelligent Student Mentor</div>',
    unsafe_allow_html=True
)


@st.cache_resource
def load_model():
    return pipeline(
        "text-generation",
        model="HuggingFaceTB/SmolLM2-360M-Instruct",
        torch_dtype=torch.float32
    )


with st.spinner("Loading AI model..."):
    chatbot = load_model()


if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.header("📚 Mentora AI")

    st.write("Choose a learning area:")

    category = st.selectbox(
        "Category",
        [
            "General",
            "Python",
            "Java",
            "Artificial Intelligence",
            "Machine Learning",
            "DBMS",
            "Data Science",
            "Project Guidance",
            "Interview Preparation"
        ]
    )

    st.divider()

    st.subheader("💡 Suggested Questions")

    if category == "Python":
        st.write("• What is Python?")
        st.write("• Explain Python loops.")
        st.write("• What is a list in Python?")

    elif category == "Java":
        st.write("• What is Java?")
        st.write("• Explain OOP concepts.")
        st.write("• What is inheritance?")

    elif category == "Artificial Intelligence":
        st.write("• What is AI?")
        st.write("• What is an AI model?")
        st.write("• Explain neural networks.")

    elif category == "Machine Learning":
        st.write("• What is machine learning?")
        st.write("• Explain supervised learning.")
        st.write("• What is classification?")

    elif category == "DBMS":
        st.write("• What is DBMS?")
        st.write("• Explain normalization.")
        st.write("• What is a primary key?")

    else:
        st.write("• Explain AI in simple words.")
        st.write("• Give me a study plan.")
        st.write("• Help me with my project.")

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


if len(st.session_state.messages) == 0:
    st.info(
        "👋 Hello! I am Mentora AI. "
        "Ask me anything about your studies, programming, "
        "AI, projects, or interview preparation."
    )


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


user_input = st.chat_input("Ask Mentora AI anything...")


if user_input:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    prompt = f"""
You are Mentora AI, a helpful and friendly student mentor.

The student's selected learning category is: {category}

Answer the student's question clearly and accurately.

Use simple language suitable for a college student.
If the question is about programming, provide a simple example when useful.
Do not make the answer unnecessarily complicated.

Student question:
{user_input}

Answer:
"""

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:
                result = chatbot(
                    prompt,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )

                generated_text = result[0]["generated_text"]

                if "Answer:" in generated_text:
                    response = generated_text.split(
                        "Answer:",
                        1
                    )[1].strip()
                else:
                    response = generated_text.strip()

                if response == "":
                    response = "Sorry, I could not generate an answer."

            except Exception as e:
                response = f"Sorry, an error occurred: {e}"

        st.markdown(response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )


st.divider()

st.caption(
    "Mentora AI | Built with Python, Streamlit, Hugging Face Transformers and PyTorch"
)
```
