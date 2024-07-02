import streamlit as st
from flask.Emotion_spotting_service import _Emotion_spotting_service
from flask.Genre_spotting_service import _Genre_spotting_service
from flask.Beat_tracking_service import _Beat_tracking_service
from diffusers import StableDiffusionPipeline
import torch

emo_list = []
gen_list = []
tempo_list = []
@st.cache_resource
def load_emo_model():
    emo_service = _Emotion_spotting_service("flask/emotion_model.h5")
    return emo_service
@st.cache_resource
def load_genre_model():
    gen_service = _Genre_spotting_service("flask/Genre_classifier_model.h5")
    return gen_service

@st.cache_resource
def load_beat_model():
    beat_service = _Beat_tracking_service()
    return beat_service

@st.cache_resource
def load_image_model():
     pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",torch_dtype=torch.float16).to("cuda")
     pipeline.load_lora_weights("Weights/pytorch_lora_weights.safetensors", weight_name="pytorch_lora_weights.safetensors")
     return pipeline


if 'emotion' not in st.session_state:
    st.session_state.emotion = None

if 'genre' not in st.session_state:
    st.session_state.genre = None

if 'beat' not in st.session_state:
    st.session_state.beat = None

emotion_service = load_emo_model()
genre_service = load_genre_model()
beat_service = load_beat_model()
image_service = load_image_model()

st.title("Music2Image webpage")
user_input = st.file_uploader("Upload your wav/mp3 files here", type=["wav","mp3"],key = "file_uploader")
st.caption("Generate images from your audio file")
st.audio(user_input)
c1,c2,c3 = st.columns([1,1,1])
with c1:
    if st.button("Generate emotion"):
        emotion = emotion_service.predict(user_input)
        st.session_state.emotion = emotion
    st.text(st.session_state.emotion)
with c2:
    if st.button("Generate genre"):
        genre = genre_service.predict(user_input)
        st.session_state.genre = genre
    st.text(st.session_state.genre)

with c3:
    if st.button("Generate beat"):
        beat = beat_service.get_beat(user_input)
        st.session_state.beat = beat
    st.text(st.session_state.beat)

if st.session_state.emotion != None and st.session_state.genre != None and st.session_state.beat != None:
    text_output = None
    if st.button("Generate text description to be fed into stable diffusion"):
        st.caption("Text description of your music file")
        text_output = "This piece of music falls under the " + st.session_state.genre[0] + " genre. It is of tempo " + str(int(st.session_state.beat)) + " and evokes a sense of" + st.session_state.emotion + "."
        st.text(text_output)
    if text_output:
        if st.button("Generate image from text description"):
            image = image_service(text_output)
            st.image(image)
