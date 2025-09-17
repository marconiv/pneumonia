import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
import tflite_runtime.interpreter as tflite

# Caminho do modelo TFLite
TFLITE_PATH = "chest_xray_model.tflite"

# Função para carregar o modelo TFLite (cacheado para não recarregar a cada uso)
@st.cache_resource
def load_tflite_model():
    interpreter = tflite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    return interpreter

# Função para pré-processar imagem
def preprocess_uploaded_image(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(180, 180), color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array, img

# Função para fazer predição com TFLite
def predict_tflite(interpreter, img_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Converte a imagem para o dtype esperado pelo modelo
    img_array = img_array.astype(input_details[0]["dtype"])

    # Passa os dados de entrada
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Roda a inferência
    interpreter.invoke()

    # Obtém a saída
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# ==================== APP ====================
st.set_page_config(page_title="Classificação de Raios-X", layout="centered")
st.title("🩺 Classificação de Raios-X de Tórax (Normal vs Pneumonia)")

# Carregar modelo
interpreter = load_tflite_model()

# Upload de imagem
uploaded_file = st.file_uploader("Envie uma imagem (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Pré-processa
    img_array, img_display = preprocess_uploaded_image(uploaded_file)

    # Predição
    prediction = predict_tflite(interpreter, img_array)[0]  # vetor de saída

    # Assumindo saída binária [Normal, Pneumonia]
    prob_normal = prediction[0]
    prob_pneumonia = prediction[1]
    label = "Pneumonia" if prob_pneumonia > prob_normal else "Normal"
    prob = max(prob_pneumonia, prob_normal)

    # Mostrar resultados
    st.image(img_display, caption=f"Imagem enviada ({label})", use_column_width=True)
    st.markdown(f"**Classe prevista:** {label}")
    st.markdown(f"**Probabilidade:** {prob:.2%}")
