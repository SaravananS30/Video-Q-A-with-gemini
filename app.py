import streamlit as st
import time
import os
import tempfile
from google import genai
from google.genai import types

# --- Page Configuration ---
st.set_page_config(page_title="Video Q&A with Gemini", page_icon="??")

st.title("Video Q&A with Gemini")

# --- Sidebar: Configuration ---
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter Google API Key", type="password")
model_name = st.sidebar.selectbox(
    "Select Model",
    ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp", "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-preview-09-2025"],
    index=0
)

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_video" not in st.session_state:
    st.session_state.processed_video = None
if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None


# --- Helper Functions ---
def process_video(uploaded_file, client):
    """
    Saves uploaded streamlit file to temp disk, uploads to Google GenAI,
    waits for processing, and returns the file object.
    """
    # 1. Save Streamlit UploadedFile to a temporary file on disk
    # Google API expects a file path, not a memory buffer
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    try:
        status_placeholder = st.empty()
        status_placeholder.info("Uploading video to Google GenAI...")

        # 2. Upload to Google Files API
        video_file = client.files.upload(file=tmp_file_path)

        # 3. Wait for processing
        status_placeholder.info("Processing video context (this may take a moment)...")
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = client.files.get(name=video_file.name)

        if video_file.state.name == "FAILED":
            status_placeholder.error("Video processing failed.")
            return None

        status_placeholder.success("Video processed and ready for analysis!")
        time.sleep(2)
        status_placeholder.empty()
        return video_file

    finally:
        # Clean up local temp file
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


# --- Main Logic ---

if api_key:
    client = genai.Client(api_key=api_key)

    # File Uploader
    uploaded_file = st.file_uploader("Upload a Video (MP4)", type=["mp4", "mov", "avi"])

    if uploaded_file:
        # Check if this is a new file or the same one
        if st.session_state.current_file_name != uploaded_file.name:
            # Reset state for new file
            st.session_state.processed_video = None
            st.session_state.chat_history = []
            st.session_state.current_file_name = uploaded_file.name

            # Upload and Process
            with st.spinner("Uploading and analyzing video..."):
                video_object = process_video(uploaded_file, client)
                st.session_state.processed_video = video_object

    # Chat Interface
    if st.session_state.processed_video:
    
        # --- New Section: Chunk/Token Inspector ---
        st.sidebar.divider()
        st.sidebar.subheader("Video Inspector")
        
        with st.sidebar.status("Analyzing Video Chunks..."):
            # 1. Get Token Count
            try:
                count_resp = client.models.count_tokens(
                    model=model_name,
                    contents=[st.session_state.processed_video]
                )
                total_tokens = count_resp.total_tokens
                st.write(f"**Total Tokens:** {total_tokens:,}")
                
                # 2. Estimate Duration based on tokens (Approximation)
                # Gemini usually uses ~263 tokens per second of video
                est_seconds = total_tokens / 263
                st.write(f"**Est. Processed Duration:** {est_seconds:.2f} seconds")
                
            except Exception as e:
                st.error(f"Could not count tokens: {e}")
    
        # 3. Debugging Prompt
        if st.sidebar.button("Check Video Vision"):
            with st.spinner("Asking model what it sees..."):
                debug_response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        st.session_state.processed_video,
                        "List the timestamps of the start and end of this video. Then describe the visual content of the very first second and the very last second."
                    ]
                )
                st.sidebar.info(debug_response.text)

        # Display Chat History
        for role, message in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(message)

        # Chat Input
        user_query = st.chat_input("Ask a question about the video...")

        if user_query:
            # 1. Display User Message
            with st.chat_message("user"):
                st.markdown(user_query)
            st.session_state.chat_history.append(("user", user_query))

            # 2. Generate Answer
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # We pass the ALREADY processed video object from session_state
                        # This acts as the "embedded" memory context
                        response = client.models.generate_content(
                            model=model_name,
                            contents=[
                                st.session_state.processed_video,
                                user_query
                            ]
                        )
                        answer = response.text
                        st.markdown(answer)
                        st.session_state.chat_history.append(("assistant", answer))
                    except Exception as e:
                        st.error(f"Error generating response: {e}")

    elif uploaded_file and not st.session_state.processed_video:
        st.warning("Video upload failed. Please try again.")
    else:
        st.info("Please upload a video to start chatting.")

else:
    st.warning("Please enter your Google API Key in the sidebar to proceed.")
