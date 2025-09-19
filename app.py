import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import requests
from io import BytesIO
import time

# ==================== CONFIG ====================
st.set_page_config(page_title="Classificação de Raios-X", layout="centered")

# Caminho do modelo TFLite
TFLITE_PATH = "chest_xray_model.tflite"

# URLs das imagens públicas no GitHub
IMAGE_URLS = [
    f"https://raw.githubusercontent.com/marconiv/pneumonia/main/samples/{i}_imagem.jpeg"
    for i in range(1, 11)
]

# ==================== HELPERS ====================
@st.cache_resource
def load_tflite_model():
    interpreter = tflite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    return interpreter

def preprocess_image(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize((180, 180))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 180, 180, 3)
    return img_array, img

def predict_tflite(interpreter, img_array: np.ndarray):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    img_array = img_array.astype(input_details[0]["dtype"])
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])
    return output_data

@st.cache_data(ttl=3600)
def fetch_image_from_url(url: str):
    r = requests.get(url, timeout=10)
    if r.status_code == 200 and "image" in r.headers.get("Content-Type", ""):
        return Image.open(BytesIO(r.content))
    return None

def expected_label_by_index(index_zero_based: int) -> str:
    # regra: ímpar = Normal, par = Pneumonia (considerando 1..10)
    one_based = index_zero_based + 1
    return "Normal" if one_based % 2 != 0 else "Pneumonia"

# ==================== UI TOP ====================
with st.spinner("Carregando aplicação... Isso pode levar alguns segundos se a aplicação estava 'dormindo'."):
    time.sleep(2)

st.info(
    "ℹ️ Se o carregamento demorar um pouco, é normal: a aplicação pode estar 'dormindo' e "
    "está sendo reativada automaticamente pelo servidor do Streamlit."
)

st.warning(
    "⚠️ Este projeto é de uso educacional e demonstrativo. "
    "Não deve ser utilizado em ambiente clínico real sem validação regulamentada."
)

st.title("🩺 Classificação de Raios-X de Tórax (Normal vs Pneumonia)")

# ==================== LOAD MODEL ====================
interpreter = load_tflite_model()

# ==================== GRID DE IMAGENS (GITHUB) ====================
st.subheader("Escolha uma imagem de amostra (GitHub)")

if "selected_url" not in st.session_state:
    st.session_state.selected_url = None

cols = st.columns(5)
for idx, url in enumerate(IMAGE_URLS):
    col = cols[idx % 5]
    try:
        thumb = fetch_image_from_url(url)
