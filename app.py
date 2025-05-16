import streamlit as st
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from PIL import Image
import io
import base64
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Gemini Image Generator (AI Studio)",
    page_icon="✨",
    layout="wide"
)

st.title("✨ Image Generator with Google AI Studio API")

st.write("""
This application generates images from text prompts using a Google Gemini model
(specifically, one capable of image generation) via the Google AI Studio API.
""")

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Configuration")
    # Use st.text_input with type="password" for API key
    api_key = st.text_input("Google AI Studio API Key", type="password", key="google_api_key_input")

    st.markdown("Or store in environment variable `GOOGLE_API_KEY` or Streamlit Secrets `st.secrets.GOOGLE_API_KEY`")

    # Prioritize key from input, then secrets, then environment
    google_api_key = api_key or st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    model_name = st.selectbox(
        "Select Gemini Model (must support image generation)",
        ["gemini-2.0-flash-preview-image-generation", "gemini-1.5-flash-latest"], # Include a common text+image model
        index=0 # Default to the explicit image generation preview model
    )

    st.markdown("""
    **Note:** Ensure the selected model supports image generation.
    `gemini-2.0-flash-preview-image-generation` is designed for this.
    Other models like `gemini-1.5-flash-latest` might generate interleaved
    text and images based on the prompt.
    """)

    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.4, step=0.1)
    top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
    top_k = st.slider("Top K", min_value=0, max_value=100, value=32, step=1)
    # Not all models support sample_count directly for image output via this API
    # Sticking to basic generation parameters

# --- Main Application Area ---
st.header("Generate Your Content") # Changed from Image to Content
prompt = st.text_area("Enter a detailed description for the image/content:", height=150) # Updated prompt text

generate_button = st.button("Generate Content")

# Store generated content in session state
if 'generated_content' not in st.session_state:
    st.session_state.generated_content = None

if generate_button and prompt:
    if not google_api_key:
        st.warning("Please enter your Google AI Studio API Key in the sidebar or set the environment variable.")
    else:
        try:
            # Configure the generativeai library with the API key
            genai.configure(api_key=google_api_key)

            # Initialize the Generative Model
            model = genai.GenerativeModel(model_name)

            # Set generation configuration - REMOVED JSON MODE PARAMETERS
            generation_config = GenerationConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                # response_mime_type='application/json', # Removed
                # response_schema=... # Removed
            )

            st.info(f"Generating content using model: `{model_name}`...")

            # Make the API call
            # For image generation, the model might return multiple parts (text and image)
            response = model.generate_content(
                prompt,
                generation_config=generation_config,
                request_options={"timeout": 600} # Increase timeout for generation
            )

            # Store the response in session state
            st.session_state.generated_content = response

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error("Please check your API key and ensure the selected model is correct and available.") # Updated error message


# --- Display Generated Content ---
if st.session_state.generated_content:
    st.subheader("Generated Content:")

    try:
        # Access the parts from the response candidate
        # Gemini models often return a list of candidates, pick the first one
        if st.session_state.generated_content.candidates:
            candidate = st.session_state.generated_content.candidates[0]
            # Check if the candidate was blocked for safety reasons
            if hasattr(candidate, 'finish_reason') and candidate.finish_reason == 'SAFETY':
                st.warning("Content generation was blocked due to safety concerns.")
                if hasattr(candidate, 'safety_ratings'):
                    st.write("Safety Ratings:", candidate.safety_ratings)
            elif hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                for i, part in enumerate(candidate.content.parts):
                    if hasattr(part, 'inline_data') and part.inline_data:
                        # This part is image data
                        try:
                            # Inline data structure has 'mime_type' and 'data'
                            if hasattr(part.inline_data, 'data') and part.inline_data.data:
                                image_bytes = base64.b64decode(part.inline_data.data)
                                image = Image.open(io.BytesIO(image_bytes))
                                st.image(image, caption=f"Generated Image {i+1}", use_column_width=True)
                            else:
                                st.warning(f"Image part {i+1} found, but no data.")
                        except Exception as img_e:
                            st.warning(f"Could not decode or display image part {i+1}: {img_e}")
                            # Optionally display raw part data for debugging
                            # st.json(part.to_dict())
                    elif hasattr(part, 'text') and part.text:
                        # This part is text data
                        st.write(f"Text Part {i+1}:")
                        st.write(part.text)
                    else:
                        st.write(f"Unknown Part {i+1}: {part}")

            # Handle cases where the model returns only text directly
            elif hasattr(st.session_state.generated_content, 'text') and st.session_state.generated_content.text:
                 st.write("Generated Text:")
                 st.write(st.session_state.generated_content.text)
            else:
                st.warning("The model generated content but it did not contain recognizable image or text parts in the expected structure.")
                # Display the full response for debugging
                st.write("Raw API Response (for debugging):")
                st.json(st.session_state.generated_content.to_dict()) # Use to_dict() for easier viewing

        else:
             st.warning("The model did not return any candidates.")
             # Check for prompt feedback or blocked reasons if available
             if hasattr(st.session_state.generated_content, 'prompt_feedback'):
                 st.write("Prompt Feedback:", st.session_state.generated_content.prompt_feedback)


    except Exception as display_e:
        st.error(f"An error occurred while displaying the results: {display_e}")
        st.write("Raw API Response (for debugging):")
        # Attempt to convert to dict safely for display
        try:
            st.json(st.session_state.generated_content.to_dict())
        except:
             st.write(st.session_state.generated_content) # Fallback if to_dict() fails


elif generate_button and not prompt:
    st.warning("Please enter a prompt to generate content.")

st.markdown("---")
st.write("Built with ❤️ using Streamlit and the Google AI Studio API.")
