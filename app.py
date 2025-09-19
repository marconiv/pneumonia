import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import requests
from io import BytesIO

# Caminho do modelo TFLite
TFLITE_PATH = "chest_xray_model.tflite"

# Função para carregar o modelo TFLite (cacheado para não recarregar a cada uso)
@st.cache_resource
def load_tflite_model():
    interpreter = tflite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    return interpreter

# Função para pré-processar imagem (em RGB, 3 canais)
def preprocess_image(img):
    img = img.convert("RGB")  # garante 3 canais
    img = img.resize((180, 180))  # redimensiona para o input do modelo
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 180, 180, 3)
    return img_array, img

# Função para fazer predição com TFLite
def predict_tflite(interpreter, img_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Ajusta dtype da entrada
    img_array = img_array.astype(input_details[0]["dtype"])

    # Passa os dados
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Roda a inferência
    interpreter.invoke()

    # Obtém a saída
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# ==================== APP ====================
st.set_page_config(page_title="Classificação de Raios-X", layout="centered")

# Aviso de uso educacional
st.warning("⚠️ Este projeto é de uso educacional e demonstrativo. "
           "Não deve ser utilizado em ambiente clínico real sem validação regulamentada.")

st.title("🩺 Classificação de Raios-X de Tórax (Normal vs Pneumonia)")

# Carregar modelo
interpreter = load_tflite_model()

# ==================== IMAGENS DO GITHUB ====================
st.subheader("Escolha uma imagem de amostra (GitHub)")

# URLs das imagens públicas no GitHub
image_urls = [
    f"https://raw.githubuse
