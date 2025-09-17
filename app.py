import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Caminho do modelo TFLite
TFLITE_PATH = "chest_xray_model.tflite"

# Função para carregar o modelo TFLite
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
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

    # Converte imagem para o dtype esperado pelo modelo
    img_array = img_array.astype(input_details[0]["dtype"])

    # Passa os dados de entrada
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Roda a inferência
    interpreter.invoke()

    # Obtém a saída
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# App Streamlit
st.title("Classificação de Raios-X de Tórax (TFLite)")

# Carregar modelo
interpreter = load_tflite_model()

# Upload de imagem
uploaded_file = st.file_uploader("Envie uma imagem (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_array, img_display = preprocess_uploaded_image(uploaded_file)
    prediction = predict_tflite(interpreter, img_array)[0]  # vetor com saída

    # Assumindo saída binária [Normal, Pneumonia]
    prob_pneumonia = prediction[1]
    prob_normal = prediction[0]
    label = "Pneumonia" if prob_pneumonia > prob_normal else "Normal"
    prob = max(prob_pneumonia, prob_normal)

    # Mostrar resultados
    st.image(img_display, caption=f"Imagem enviada ({label})", use_column_width=True)
    st.write(f"**Classe prevista:** {label}")
    st.write(f"**Probabilidade:** {prob:.2%}")
