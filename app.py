# streamlit_app.py

import streamlit as st
import os
import base64
import mimetypes
from google import genai
from google.genai import types

# Initialize Gemini client
genai.Client(
api_key=os.environ.get("GEMINI_API_KEY"),
)

def save_binary_file(file_name, data):
    with open(file_name, "wb") as f:
        f.write(data)
    return file_name

def generate_image(prompt, file_name):
    client = genai.Client()
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
        response_modalities=["IMAGE", "TEXT"],
        response_mime_type="text/plain",
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if (
            chunk.candidates
            and chunk.candidates[0].content
            and chunk.candidates[0].content.parts
            and chunk.candidates[0].content.parts[0].inline_data
        ):
            inline_data = chunk.candidates[0].content.parts[0].inline_data
            file_extension = mimetypes.guess_extension(inline_data.mime_type)
            saved_path = save_binary_file(f"{file_name}{file_extension}", inline_data.data)
            return saved_path

    return None

# Streamlit UI
st.title("Gemini Image Generator")

prompt = st.text_area("Enter a prompt for the image")
file_name = st.text_input("Enter output file name (without extension)", value="generated_image")

if st.button("Generate Image"):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating image..."):
            image_path = generate_image(prompt, file_name)
            if image_path:
                st.success(f"Image saved: {image_path}")
                st.image(image_path)
            else:
                st.error("Image generation failed.")
