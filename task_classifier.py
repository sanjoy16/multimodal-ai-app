import os
import requests
import base64
from io import BytesIO
from PIL import Image
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from audio_recorder_streamlit import audio_recorder  # 🎙️ for mic input
import wave   

# =====================
# Load environment variables
# =====================
load_dotenv()
google_api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
pollinations_token = st.secrets.get("POLLINATIONS_TOKEN", os.getenv("POLLINATIONS_TOKEN"))
gladia_api_key = st.secrets.get("GLADIA_API_KEY", os.getenv("GLADIA_API_KEY"))

if not google_api_key:
    st.error("❌ GOOGLE_API_KEY not found! Please set it in .env or Streamlit Secrets.")
    st.stop()

if not pollinations_token:
    st.error("❌ POLLINATIONS_TOKEN not found! Please set it in .env or Streamlit Secrets.")
    st.stop()

if not gladia_api_key:
    st.error("❌ GLADIA_API_KEY not found! Please set it in .env or Streamlit Secrets.")
    st.stop()

# =====================
# Initialize Gemini LLM
# =====================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    google_api_key=google_api_key,
    streaming=True
)

# =====================
# Streamlit App Layout
# =====================
st.set_page_config(page_title="🤖Sanjoy, Multimodal AI App", page_icon="🤖", layout="centered")
st.title("🤖 Multimodal AI App (Text + Image Generator + Image Q&A + Voice Input)")

tab1, tab2, tab3 = st.tabs(["💬 Text & Voice Chat", "🎨 Image Generator", "🖼️ Image Q&A"])

# =====================
# TEXT + VOICE CHAT TAB
# =====================
with tab1:
    st.subheader("⚡ Fast Text + Voice Task Classifier & Gemini Chat")

    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    def compose_prompt(conversation, current_query):
        prompt_text = ""
        for user_q, assistant_a in conversation:
            prompt_text += f"User: {user_q}\nAssistant: {assistant_a}\n"
        prompt_text += f"User: {current_query}\nAssistant:"
        return prompt_text

    def handle_text_task(conversation, query: str):
        lower_q = query.lower()
        if any(phrase in lower_q for phrase in [
            "who built this agent", "who created this agent", "who made this agent",
            "who is the developer", "who is the creator"
        ]):
            return "This agent was built by **Sanjoy Bhattacharjee**."
    
        prompt = compose_prompt(conversation, query)
    
        response_placeholder = st.empty()
        final_response = ""
    
        # ⚡ Faster streaming loop (optimized)
        with st.spinner("⚡ Generating response..."):
            for chunk in llm.stream(prompt):
                if chunk.content:
                    final_response += chunk.content
                    # Faster updates using write() instead of markdown
                    response_placeholder.write(f"**Answer (streaming):**\n\n{final_response}")
            # After streaming ends, show final formatted text
            response_placeholder.markdown(f"**✅ Final Answer:**\n\n{final_response}")
    
        return final_response

    # =====================
    # 🎙️ Voice Input + Text Input Section
    # =====================
    query = st.text_input("💬 Enter your request:", key="input_query")

    # Record voice (audio_recorder creates mic button)
    st.write("🎙️ Speak your query below (up to 30 seconds):")
    audio_bytes = audio_recorder(
        text="Click to start/stop recording",
        recording_color="#FF4B4B",
        neutral_color="#4B9EFF",
        icon_size="2x",  # Larger mic button
        energy_threshold=(-1.0, 1.0)
    )

    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        tmp_path = "temp_audio.wav"
        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)

        with wave.open(tmp_path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            duration = frames / float(rate)

        if duration < 2:
            st.warning("⚠️ Your recording was too short — please record for at least 2 seconds.")
        else:
            with st.spinner("🎧 Transcribing your voice..."):
                files = {'audio': ("voice.wav", audio_bytes, "audio/wav")}
                headers = {"x-gladia-key": gladia_api_key}
                response = requests.post(
                    "https://api.gladia.io/audio/text/audio-transcription/",
                    headers=headers,
                    files=files
                )

                if response.status_code == 200:
                    result_json = response.json()
                    text_result = ""

                    # ✅ SAFELY extract transcription
                    if isinstance(result_json, dict):
                        if "transcription" in result_json:
                            text_result = result_json["transcription"]
                        elif "result" in result_json and isinstance(result_json["result"], list):
                            for item in result_json["result"]:
                                if isinstance(item, dict) and "transcription" in item:
                                    text_result = item["transcription"]
                                    break
                        elif "prediction" in result_json:
                            text_result = result_json["prediction"]
                    elif isinstance(result_json, list):
                        for item in result_json:
                            if isinstance(item, dict) and "transcription" in item:
                                text_result = item["transcription"]
                                break

                    # ✅ Normalize to string
                    if isinstance(text_result, list):
                        text_result = " ".join(str(x) for x in text_result)
                    if isinstance(text_result, dict):
                        text_result = text_result.get("transcription", "")

                    if isinstance(text_result, str) and text_result.strip():
                        st.success(f"🗣️ You said: {text_result}")
                        query = text_result.strip()
                        ans = handle_text_task(st.session_state.conversation, query)
                        st.session_state.conversation.append((query, ans))
                    else:
                        st.warning("⚠️ Couldn't extract valid text from the API response.")
                else:
                    st.error(f"❌ Gladia API Error: {response.text}")

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
            st.code(assistant_a, language="markdown")
            st.markdown("---")

# =====================
# IMAGE GENERATOR TAB
# =====================
with tab2:
    st.subheader("🎨 Pollinations.AI Free Image Generator")

    img_prompt = st.text_input("📝 Enter your image prompt:", key="img_prompt")

    styles = ["Realistic", "Cartoon", "Fantasy", "Minimalist"]
    selected_style = st.radio("🎨 Choose a style:", styles, horizontal=True)

    def smart_enhance_prompt(user_prompt, style):
        quick_prompt = f"Rewrite this short prompt into a detailed {style} image generation description: {user_prompt}"
        response = llm.invoke(quick_prompt)
        return response.content.strip()

    @st.cache_data(show_spinner=False)
    def fetch_image(final_prompt, token):
        url = f"https://image.pollinations.ai/prompt/{final_prompt}?token={token}"
        return requests.get(url).content

    if st.button("Generate Image"):
        if not img_prompt:
            st.warning("⚠️ Please enter a prompt before generating an image.")
        else:
            with st.spinner(f"🎨 Generating {selected_style} image..."):
                final_prompt = smart_enhance_prompt(img_prompt, selected_style)
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
# IMAGE Q&A TAB
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

