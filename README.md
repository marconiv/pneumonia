🩺 Classificação de Raios-X de Tórax com IA

Este projeto utiliza Redes Neurais Convolucionais (CNNs) para classificar exames de raios-X de tórax em duas categorias: Normal e Pneumonia.
A aplicação foi desenvolvida em Streamlit e está disponível online para demonstração interativa.

🚀 Demonstração Online

👉 Acesse a aplicação no Streamlit Cloud : https://pneumonia-portfolio-marconi-vieira-infochoice.streamlit.app/

📂 Estrutura do Projeto

app.py → Código principal da aplicação (Streamlit).

chest_xray_model.tflite → Modelo treinado convertido para TensorFlow Lite (leve, ~13 MB).

requirements.txt → Dependências necessárias para execução.

examples/ → Imagens de exemplo (opcional para teste rápido).

⚙️ Como Funciona

O usuário envia uma imagem (JPG/PNG) de raio-X de tórax.

A imagem é pré-processada (grayscale, 180×180 px, normalização).

O modelo TFLite roda a predição em tempo real.

O resultado é exibido na tela com:

Classe prevista (Normal ou Pneumonia)

Probabilidade da predição

Imagem original carregada

🛠️ Execução Local
1. Clone este repositório
git clone https://github.com/seuusuario/seuprojeto.git
cd seuprojeto

2. Crie um ambiente virtual (opcional, mas recomendado)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

3. Instale as dependências
pip install -r requirements.txt

4. Execute a aplicação
streamlit run app.py


Acesse no navegador em: http://localhost:8501

📊 Dataset Utilizado

O modelo foi treinado no dataset público Chest X-Ray Images (Pneumonia) disponível no Kaggle
.

📌 Tecnologias e Bibliotecas

Streamlit

TensorFlow / TFLite

NumPy

Matplotlib

scikit-image

OpenCV

📜 Licença

Este projeto é de uso educacional e demonstrativo.
Não deve ser utilizado em ambiente clínico real sem validação regulamentada.

👨‍💻 Autor

Marconi Vieira
📌 Portfólio: marconivieira.com.br
