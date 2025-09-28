import os
import requests
import base64
from io import BytesIO
from PIL import Image
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

# =====================
# Load environment variables
# =====================
load_dotenv()
google_api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
openai_api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
pollinations_token = st.secrets.get("POLLINATIONS_TOKEN", os.getenv("POLLINATIONS_TOKEN"))

if not google_api_key:
    st.error("❌ GOOGLE_API_KEY not found! Please set it in .env or Streamlit Secrets.")
    st.stop()

if not openai_api_key:
    st.error("❌ OPENAI_API_KEY not found! Please set it in .env or Streamlit Secrets.")
    st.stop()

if not pollinations_token:
    st.error("❌ POLLINATIONS_TOKEN not found! Please set it in .env or Streamlit Secrets.")
    st.stop()

# =====================
# Initialize LLMs
# =====================
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=google_api_key,
    streaming=True
)

openai_llm = ChatOpenAI(
    model="gpt-5-nano",  # ⚡ You can switch between gpt-4o, gpt-4o-mini, gpt-3.5
    temperature=0,
    api_key=openai_api_key,
    streaming=True
)

# =====================
# Routing logic for LLM selection
# =====================
def choose_llm(query: str):
    q = query.lower()
    if "image" in q or "analyze" in q:
        return gemini_llm, "Gemini", "Gemini is better at multimodal reasoning (text + images).", 0.92
    elif "summarize" in q or "explain" in q:
        return openai_llm, "OpenAI gpt-5-nano", "GPT is excellent for summarization and structured text tasks.", 0.95
    elif "creative" in q or "story" in q:
        return openai_llm, "OpenAI gpt-5-nano", "GPT is strong for creative writing tasks.", 0.93
    else:
        return gemini_llm, "Gemini", "Gemini flash is faster for general-purpose queries.", 0.90

# =====================
# TEXT CHAT TAB
# =====================
with tab1:
    st.subheader("⚡ Dynamic LLM Router (Gemini + OpenAI)")

    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    def handle_text_task(conversation, query: str):
        # Choose which LLM to use
        llm, name, reason, accuracy = choose_llm(query)

        prompt = f"User: {query}\nAssistant:"
        response_placeholder = st.empty()
        final_response = ""

        for chunk in llm.stream(prompt):
            final_response += chunk.content or ""
            response_placeholder.markdown(f"**Answer (streaming):**\n\n{final_response}")

        # Append meta info about LLM choice
        meta_info = f"\n\n---\n**ℹ️ Model Used:** {name}\n**Reason:** {reason}\n**Estimated Accuracy:** {accuracy*100:.1f}%"
        final_response += meta_info

        return final_response

    query = st.text_input("💬 Enter your request:", key="input_query")

    col1, col2 = st.columns([1, 3])
    with col1:
        process_clicked = st.button("Process")
    with col2:
        clear_clicked = st.button("Clear Conversation")

    if clear_clicked:
        st.session_state.conversation = []

    if process_clicked and query:
        ans = handle_text_task(st.session_state.conversation, query)
        st.session_state.conversation.append((query, ans))

    if st.session_state.conversation:
        st.markdown("### 🗂️ Conversation History")
        for user_q, assistant_a in st.session_state.conversation:
            st.markdown(f"**User:** {user_q}")
            st.markdown(assistant_a)
            st.markdown("---")

# =====================
# IMAGE GENERATOR TAB
# =====================# =====================
# IMAGE GENERATOR TAB (with auto-enhancement + styles)
# =====================
# =====================
# IMAGE GENERATOR TAB (Gemini-enhanced + Faster Caching)
# =====================
with tab2:
    st.subheader("🎨 Pollinations.AI Free Image Generator")

    img_prompt = st.text_input("📝 Enter your image prompt:", key="img_prompt")

    # Style options
    styles = ["Realistic", "Cartoon", "Fantasy", "Minimalist"]
    selected_style = st.radio("🎨 Choose a style:", styles, horizontal=True)

    # Function: Ask Gemini to expand + improve the prompt
    def smart_enhance_prompt(user_prompt, style):
        quick_prompt = f"Rewrite this short prompt into a detailed {style} image generation description: {user_prompt}"
        response = llm.invoke(quick_prompt)  # using Gemini directly
        return response.content.strip()

    # Function: cache Pollinations image fetch for speed
    @st.cache_data(show_spinner=False)
    def fetch_image(final_prompt, token):
        url = f"https://image.pollinations.ai/prompt/{final_prompt}?token={token}"
        return requests.get(url).content

    if st.button("Generate Image"):
        if not img_prompt:
            st.warning("⚠️ Please enter a prompt before generating an image.")
        else:
            with st.spinner(f"🎨 Generating {selected_style} image..."):
                # Auto-enhance the prompt using Gemini
                final_prompt = smart_enhance_prompt(img_prompt, selected_style)

                # Fetch image (cached if repeated)
                try:
                    img_bytes = fetch_image(final_prompt, pollinations_token)
                    img = Image.open(BytesIO(img_bytes))
                    st.image(img, caption=final_prompt)

                    buf = BytesIO()
                    img.save(buf, format="PNG")
                    st.download_button(
                        label="📥 Download Image",
                        data=buf.getvalue(),
                        file_name="pollinations_image.png",
                        mime="image/png"
                    )
                except Exception as e:
                    st.error(f"❌ Failed to generate image: {e}")


# =====================
# IMAGE Q&A TAB (FIXED with base64 encoding)
# =====================
with tab3:
    st.subheader("🖼️ Upload an Image & Ask Gemini")

    uploaded_img = st.file_uploader("📂 Upload an image", type=["jpg", "jpeg", "png"])
    qna_prompt = st.text_input("💬 Ask something about the uploaded image:")

    if st.button("Analyze Image"):
        if not uploaded_img:
            st.warning("⚠️ Please upload an image first.")
        elif not qna_prompt:
            st.warning("⚠️ Please enter a question about the image.")
        else:
            with st.spinner("🔎 Analyzing image..."):
                # ✅ Proper base64 encoding
                img_bytes = uploaded_img.read()
                img_base64 = base64.b64encode(img_bytes).decode("utf-8")
                data_url = f"data:image/png;base64,{img_base64}"

                content = [
                    {"type": "text", "text": qna_prompt},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]

                response_placeholder = st.empty()
                final_response = ""

                for chunk in llm.stream([HumanMessage(content=content)]):
                    final_response += chunk.content or ""
                    response_placeholder.markdown(f"**Answer (streaming):**\n\n{final_response}")  



