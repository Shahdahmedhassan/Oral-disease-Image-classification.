import json
from datetime import datetime

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Dental Disease Image Classification",
    page_icon="🦷",
    layout="centered",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "best_model_final.keras"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = (224, 224)  # must match the size used during training


# ---------------------------------------------------------
# Light custom styling for a more polished look
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .result-card {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        background-color: #0f766e0d;
        border: 1px solid #0f766e33;
        margin-top: 1rem;
    }
    .disclaimer {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-top: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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
# Sidebar: app info
# ---------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "This tool uses a deep learning model trained on oral/dental images "
        "to classify common oral conditions."
    )
    st.markdown("**Detected classes:**")

model, idx_to_class = load_model()

with st.sidebar:
    for class_name in idx_to_class.values():
        st.markdown(f"- {class_name}")
    st.divider()
    st.caption(
        "This tool is for educational/demo purposes only and is not a "
        "substitute for professional medical diagnosis."
    )


# ---------------------------------------------------------
# Main UI
# ---------------------------------------------------------
st.markdown('<div class="main-title">🦷 Dental Disease Image Classification</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload a photo of the mouth/teeth and the model '
    'will predict the most likely condition.</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose an image (jpg / jpeg / png)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="Uploaded image", use_column_width=True)

    with st.spinner("Analyzing image..."):
        input_arr = preprocess_image(image)
        preds = model.predict(input_arr)[0]  # array of probabilities, one per class

    # Get the class with the highest probability
    top_idx = int(np.argmax(preds))
    top_class = idx_to_class[top_idx]
    top_conf = float(preds[top_idx]) * 100

    with col2:
        st.markdown(
            f"""
            <div class="result-card">
                <div style="font-size:0.9rem;color:#6b7280;">Predicted condition</div>
                <div style="font-size:1.6rem;font-weight:700;color:#0f766e;">{top_class}</div>
                <div style="font-size:0.9rem;color:#6b7280;margin-top:0.4rem;">
                    Confidence: <strong>{top_conf:.1f}%</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Probability breakdown")
    probs_dict = {idx_to_class[i]: float(preds[i]) * 100 for i in range(len(preds))}
    probs_sorted = dict(sorted(probs_dict.items(), key=lambda x: x[1], reverse=True))
    st.bar_chart(probs_sorted)

    # -------------------------------------------------------
    # Diagnosis report section
    # -------------------------------------------------------
    st.subheader("📋 Diagnosis Report")

    scan_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    top3 = list(probs_sorted.items())[:3]

    report_html = f"""
    <div class="result-card">
        <div style="display:flex;justify-content:space-between;flex-wrap:wrap;">
            <div><strong>File name:</strong> {uploaded_file.name}</div>
            <div><strong>Date/Time:</strong> {scan_time}</div>
        </div>
        <hr style="border-color:#0f766e33;margin:0.8rem 0;">
        <div><strong>Primary finding:</strong> {top_class} ({top_conf:.1f}% confidence)</div>
        <div style="margin-top:0.6rem;"><strong>Top 3 possibilities:</strong></div>
        <ul>
            {''.join(f"<li>{name}: {conf:.1f}%</li>" for name, conf in top3)}
        </ul>
    </div>
    """
    st.markdown(report_html, unsafe_allow_html=True)

    # Plain-text version of the report, available for download
    report_text = (
        "DENTAL DISEASE IMAGE CLASSIFICATION - DIAGNOSIS REPORT\n"
        "========================================================\n"
        f"File name       : {uploaded_file.name}\n"
        f"Date/Time       : {scan_time}\n"
        f"Primary finding : {top_class} ({top_conf:.1f}% confidence)\n\n"
        "Top 3 possibilities:\n"
        + "\n".join(f"  - {name}: {conf:.1f}%" for name, conf in top3)
        + "\n\nFull probability breakdown:\n"
        + "\n".join(f"  - {name}: {conf:.1f}%" for name, conf in probs_sorted.items())
        + "\n\nDisclaimer: This report is generated by an automated model for "
        "educational/demo purposes only and is not a substitute for diagnosis "
        "by a qualified healthcare professional.\n"
    )

    st.download_button(
        label="⬇️ Download report (.txt)",
        data=report_text,
        file_name=f"dental_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
    )

    st.markdown(
        '<div class="disclaimer">⚠️ This tool is for educational/demo purposes only '
        "and is not a substitute for diagnosis by a qualified healthcare professional.</div>",
        unsafe_allow_html=True,
    )
else:
    st.info("Please upload an image to start the analysis.")
