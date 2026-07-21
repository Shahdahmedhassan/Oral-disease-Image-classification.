import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(page_title="Oral Disease Classifier", page_icon="🦷", layout="centered")

MODEL_PATH = "best_model_final.keras"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = (224, 224)  # must match the size used during training


# ---------------------------------------------------------
# Load model + class mapping once, then cache them
# (cache_resource keeps the model in memory across reruns/users)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        idx_to_class = json.load(f)
    # JSON keys are always strings, convert them back to int
    idx_to_class = {int(k): v for k, v in idx_to_class.items()}
    return model, idx_to_class


# ---------------------------------------------------------
# Preprocess the uploaded image exactly like the training generator
# (resize to 224x224, rescale pixels to 0-1, add batch dimension)
# ---------------------------------------------------------
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    arr = np.array(image).astype("float32") / 255.0  # same rescale=1/255 as training
    arr = np.expand_dims(arr, axis=0)  # shape becomes (1, 224, 224, 3)
    return arr


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.title("🦷 Oral Disease Image Classifier")
st.write("ارفع صورة للفم/الأسنان وهيقولك الموديل الحالة المتوقعة من بين الأمراض دي.")

model, idx_to_class = load_model()

uploaded_file = st.file_uploader("اختار صورة (jpg / jpeg / png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="الصورة اللي اترفعت", use_container_width=True)

    with st.spinner("جاري تحليل الصورة..."):
        input_arr = preprocess_image(image)
        preds = model.predict(input_arr)[0]  # array of probabilities, one per class

    # Get the class with the highest probability
    top_idx = int(np.argmax(preds))
    top_class = idx_to_class[top_idx]
    top_conf = float(preds[top_idx]) * 100

    st.success(f"**النتيجة المتوقعة: {top_class}**  (ثقة: {top_conf:.2f}%)")

    # Show probability breakdown for all classes
    st.subheader("تفاصيل الاحتمالات لكل كلاس")
    probs_dict = {idx_to_class[i]: float(preds[i]) * 100 for i in range(len(preds))}
    probs_sorted = dict(sorted(probs_dict.items(), key=lambda x: x[1], reverse=True))
    st.bar_chart(probs_sorted)

    st.caption("⚠️ الأداة دي لأغراض تعليمية/تجريبية فقط، ومش بديل عن تشخيص طبيب مختص.")
else:
    st.info("لسه محتاج ترفع صورة عشان يبدأ التحليل.")
