import base64
import mimetypes
import os
import time
import streamlit as st
from google import genai
from google.genai import types

# Page configuration
st.set_page_config(
    page_title="Gemini Image Generator",
    page_icon="🎨",
    layout="wide"
)

# App title and description
st.title("🎨 Gemini Image Generator")
st.markdown("Generate images using Google's Gemini 2.0 Flash model")

# API key handling
api_key = st.sidebar.text_input("Enter your Gemini API Key", type="password")
if api_key:
    os.environ["GEMINI_API_KEY"] = api_key

# Input area
with st.form("generation_form"):
    prompt = st.text_area("Enter your image prompt", 
                      placeholder="A watercolor painting of a coastal lighthouse at sunset with stormy waves crashing against the rocks",
                      height=100)
    
    # File name input
    file_name = st.text_input("Output file name (without extension)", 
                             value="gemini_image", 
                             placeholder="Enter file name without extension")
    
    # Generation button
    submit_button = st.form_submit_button("Generate Image")

# Function to save binary file
def save_binary_file(file_name, data):
    f = open(file_name, "wb")
    f.write(data)
    f.close()
    return file_name

# Image generation function
def generate_image(prompt, file_name):
    if not os.environ.get("GEMINI_API_KEY"):
        st.error("Please enter your Gemini API key in the sidebar")
        return None

    # Create a client
    try:
        client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY"),
        )
    except Exception as e:
        st.error(f"Error initializing client: {str(e)}")
        return None

    # Set up the model and content
    model = "gemini-2.0-flash-preview-image-generation"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt),
            ],
        ),
    ]
    
    generate_content_config = types.GenerateContentConfig(
        response_modalities=[
            "IMAGE",
            "TEXT",
        ],
        response_mime_type="text/plain",
    )

    # Show generating message
    with st.status("Generating image...", expanded=True) as status:
        try:
            saved_file = None
            response_text = ""
            
            # Stream the generation
            for chunk in client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=generate_content_config,
            ):
                if (
                    chunk.candidates is None
                    or chunk.candidates[0].content is None
                    or chunk.candidates[0].content.parts is None
                ):
                    continue
                
                # Handle image data
                if chunk.candidates[0].content.parts[0].inline_data:
                    st.write("Image generated! Processing...")
                    inline_data = chunk.candidates[0].content.parts[0].inline_data
                    data_buffer = inline_data.data
                    file_extension = mimetypes.guess_extension(inline_data.mime_type)
                    full_file_name = f"{file_name}{file_extension}"
                    saved_file = save_binary_file(full_file_name, data_buffer)
                    st.write(f"Image saved as: {full_file_name}")
                
                # Handle text response
                else:
                    chunk_text = chunk.text
                    if chunk_text:
                        response_text += chunk_text
                        st.write(chunk_text)
            
            status.update(label="Generation complete!", state="complete")
            return saved_file, response_text
            
        except Exception as e:
            st.error(f"Error during generation: {str(e)}")
            status.update(label="Generation failed", state="error")
            return None, str(e)

# Handle form submission
if submit_button:
    if not prompt:
        st.warning("Please enter a prompt")
    else:
        # Generate the image
        st.subheader("Generation Results")
        file_path, response_text = generate_image(prompt, file_name)
        
        # Display the generated image
        if file_path and os.path.exists(file_path):
            st.subheader("Generated Image")
            st.image(file_path, caption=prompt)
            
            # Add download button
            with open(file_path, "rb") as file:
                btn = st.download_button(
                    label="Download Image",
                    data=file,
                    file_name=os.path.basename(file_path),
                    mime=mimetypes.guess_type(file_path)[0]
                )

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
This app uses Google's Gemini 2.0 Flash model for image generation.
You need a valid Gemini API key to use this application.
""")
