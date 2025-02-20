import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

# Fungsi untuk memuat model dengan caching agar tidak perlu dimuat ulang setiap kali
@st.cache_resource(show_spinner=False)
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    if torch.cuda.is_available():
        # Gunakan GPU dengan float16 untuk performa yang lebih baik
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
    else:
        # Jika GPU tidak tersedia, muat model tanpa torch_dtype agar default float32 digunakan
        pipe = StableDiffusionPipeline.from_pretrained(model_id)
        pipe = pipe.to("cpu")
    return pipe

st.title("Image Generator")

# Input prompt dari user
prompt = st.text_input("Masukkan deskripsi gambar", "Pemandangan kota futuristik saat matahari terbenam")

# Tombol untuk menghasilkan gambar
if st.button("Hasilkan Gambar"):
    model = load_model()
    with st.spinner("Sedang menghasilkan gambar..."):
        # Menghasilkan gambar dari prompt yang diberikan
        result = model(prompt)
        image = result.images[0]
    st.image(image, caption="Gambar yang dihasilkan", use_container_width=True)
