import streamlit as st
import os
import base64
import torch
from PIL import Image
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from groq import Groq

# Set the API key for Groq
os.environ["GROQ_API_KEY"] = "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"  # Replace with your Groq API Key

# Function to get the Groq client
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    return Groq(api_key=api_key)

# Streamlit page configuration
st.title("Image Description Refinement and Generation")
st.write("Upload an image, refine the description, and generate a new image.")

# Load the model once
@st.cache_resource
def load_diffusion_model():
    if torch.cuda.is_available():
        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
    else:
        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    return pipe

pipe = load_diffusion_model()

# Step 1: Image upload by user
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Step 2: Show the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert the uploaded image to base64
    image_path = uploaded_file.name
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Step 3: Encode the image to base64
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

    # Step 4: Get image description from Groq
    client = get_groq_client()
    completion = client.chat.completions.create(
        model="llava-v1.5-7b-4096-preview",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "create a description of the image provided. My use case is I want to send this description to other text-to-image models, so the description should be accurate and contain no unnecessary information."
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}
                }
            ]
        }],
        temperature=0,
        max_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )

    # Display the generated description
    initial_description = completion.choices[0].message.content
    st.write("### Initial Description:")
    st.write(initial_description)

    # Step 5: User input for refinement
    refinement_prompt = st.text_input("Enter a prompt to refine the description:", "")

    if refinement_prompt:
        # Step 6: Refining the description using the Llama model
        lama_completion = client.chat.completions.create(
            model="llama3-groq-70b-8192-tool-use-preview",
            messages=[{
                "role": "user",
                "content": f"Refine the following description based on the prompt:\n\nDescription: {initial_description}\n\nPrompt: {refinement_prompt}"
            }],
            temperature=0,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )

        refined_description = lama_completion.choices[0].message.content
        st.write("### Refined Description:")
        st.write(refined_description)

        # Step 7: Use refined description to generate image
        st.write("Generating an image based on the refined description...")

        # Generate the image
        generated_image = pipe(refined_description).images[0]

        # Display generated image directly in Streamlit
        st.image(generated_image, caption="Generated Image", use_column_width=True)
        st.write("Image generated successfully!")
