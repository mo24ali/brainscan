"""BrainScan AI - Interface Streamlit de detection de tumeur cerebrale.

Application web permettant a l'utilisateur de televerser une IRM cerebrale et
d'obtenir la prediction du modele CNN (4 classes : glioma, meningioma, notumor,
pituitary). Les pretraitements appliques sont exactement ceux de l'entrainement :
    cv2.imread -> BGR->RGB -> cv2.resize(224x224) -> normalisation /255.

Lancement :
    streamlit run app.py
"""

import json
import os

import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "artifacts", "models")
MODEL_PATH = os.path.join(MODELS_DIR, "brainscan_model.h5")
MAPPING_PATH = os.path.join(MODELS_DIR, "class_mapping.json")
IMAGE_SIZE = 224
EXTS = ["jpg", "jpeg", "png", "bmp"]

st.set_page_config(page_title="BrainScan AI", layout="centered")


# ---------------------------------------------------------------------------
# Chargement du modele et du mapping (mis en cache)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    """Charge le modele exporte et le mapping classe -> index (une seule fois)."""
    if not os.path.exists(MODEL_PATH):
        return None, None
    model = load_model(MODEL_PATH)
    with open(MAPPING_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    class_names = [None] * len(mapping)
    for name, idx in mapping.items():
        class_names[int(idx)] = name
    return model, class_names


model, class_names = load_assets()

# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------
st.title("BrainScan AI - Detection de tumeur cerebrale")
st.caption(
    "Classification d'IRM cerebrales en 4 classes : "
    "glioma, meningioma, notumor, pituitary."
)

if model is None:
    st.error(
        "Modele introuvable. Entrainez d'abord le modele :\n\n"
        "1. `pip install -r requirements.txt`\n"
        "2. executez `notebooks/02_cnn.ipynb`\n\n"
        f"Fichier attendu : `{os.path.relpath(MODEL_PATH, BASE_DIR)}`"
    )
    st.stop()

uploaded = st.file_uploader(
    "Televersez une image IRM (.jpg, .jpeg, .png, .bmp)", type=EXTS
)

if uploaded is not None:
    # Lecture de l'image televersee en memoire -> image BGR (comme cv2.imread)
    file_bytes = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("Image illisible ou corrompue.")
        st.stop()

    # Pretraitements identiques a l'entrainement
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_resized.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # dimension batch

    # Prediction
    probabilities = model.predict(img_array, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])
    label = class_names[predicted_index]

    # Affichage
    col_img, col_pred = st.columns([1, 1])
    with col_img:
        st.image(img_rgb, caption="IRM televersee", use_container_width=True)
    with col_pred:
        if label == "notumor":
            st.success(f"Aucune tumeur detectee\n\nConfiance : {confidence * 100:.2f} %")
        else:
            st.error(f"Tumeur detectee : **{label}**\n\nConfiance : {confidence * 100:.2f} %")

        st.metric("Classe predite", label)
        st.metric("Confiance", f"{confidence * 100:.2f} %")

    st.subheader("Probabilites par classe")
    st.bar_chart({name: float(probabilities[i]) for i, name in enumerate(class_names)})

    with st.expander("Details"):
        for i, name in enumerate(class_names):
            st.write(f"- **{name}** : {probabilities[i] * 100:.2f} %")
        st.caption(
            "Pretraitements : cv2.resize 224x224 + normalisation /255 "
            "(identiques a l'entrainement)."
        )
