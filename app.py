# ==========================================
# IMPORT LIBRARIES
# ==========================================

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Logo Detection System",
    page_icon="🎯",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main {
    background-color: #0f172a;
}

.title {
    text-align:center;
    color:white;
    font-size:45px;
    font-weight:bold;
}

.subtitle {
    text-align:center;
    color:#cbd5e1;
    font-size:18px;
}

.offer-box {
    background:#ef4444;
    padding:20px;
    border-radius:15px;
    color:white;
    text-align:center;
    font-size:24px;
    font-weight:bold;
}

.result-box {
    background:#1e293b;
    padding:20px;
    border-radius:15px;
    color:white;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL
# ==========================================

MODEL_PATH = "runs/classify/train/weights/best.pt"

model = YOLO(MODEL_PATH)

# ==========================================
# OFFERS
# ==========================================

OFFERS = {

    "kfc": "🍗 50% OFF ON ALL COMBOS TODAY",

    "mcd": "🍔 BUY 1 BURGER GET 1 FREE",

    "vw": "🚗 1 DAY FREE TEST DRIVE",

    "bata": "👟 50% OFF ON FOOTWEAR",

    "bmw": "🚘 NEW BMW MODEL LAUNCHED",

    "dominos": "🍕 NEW CHEESE BURST PIZZA",

    "fastrack": "⌚ 20% OFF ON WATCHES",

    "haldiram": "🍬 BUY 1 GET 1 FREE",

    "starbuks": "☕ BUY 2 COFFEE GET 1 FREE",

    "zudio": "👕 10% OFF ON CLOTHES"

}

# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<p class="title">🎯 AI Logo Detection & Offer System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Upload Brand Logo and Get Instant Offer</p>',
    unsafe_allow_html=True
)

st.write("")

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("📋 Project Menu")

st.sidebar.success("AI Logo Classification")

st.sidebar.info("""
Supported Brands

✅ KFC

✅ McDonalds

✅ Volkswagen

✅ Bata

✅ BMW

✅ Dominos

✅ Fastrack

✅ Haldiram

✅ Starbucks

✅ Zudio
""")

# ==========================================
# CAMERA INSTRUCTIONS
# ==========================================

st.info("""
📷 Camera Usage

1. Open mobile or laptop camera

2. Capture logo image

3. Upload image below

4. AI will detect logo

5. Offer will be displayed automatically
""")

# ==========================================
# FILE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "📁 Upload Logo Image",
    type=["jpg", "jpeg", "png", "webp"]
)

# ==========================================
# PREDICTION
# ==========================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Uploaded Logo",
            use_container_width=True
        )

    # Save temp image

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    ) as tmp:

        image.save(tmp.name)

        temp_path = tmp.name

    # Prediction

    results = model.predict(
        source=temp_path,
        imgsz=224,
        verbose=False
    )

    r = results[0]

    pred_id = int(r.probs.top1)

    confidence = float(
        r.probs.top1conf
    )

    pred_name = r.names[pred_id]

    offer = OFFERS.get(
        pred_name.lower(),
        "WELCOME"
    )

    with col2:

        st.markdown("""
        <div class="result-box">
        <h2>Detection Result</h2>
        </div>
        """,
        unsafe_allow_html=True)

        st.success(
            f"Detected Logo : {pred_name.upper()}"
        )

        st.metric(
            "Confidence Score",
            f"{confidence*100:.2f}%"
        )

        st.progress(
            int(confidence*100)
        )

        st.markdown(
            f"""
            <div class="offer-box">
            {offer}
            </div>
            """,
            unsafe_allow_html=True
        )

    os.remove(temp_path)

# ==========================================
# BRAND CARDS
# ==========================================

st.write("")
st.write("---")

st.subheader("🏢 Supported Brands")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.success("KFC")
    st.success("BMW")

with c2:
    st.success("MCD")
    st.success("BATA")

with c3:
    st.success("VW")
    st.success("DOMINOS")

with c4:
    st.success("FASTRACK")
    st.success("HALDIRAM")

with c5:
    st.success("STARBUCKS")
    st.success("ZUDIO")

# ==========================================
# FOOTER
# ==========================================

st.write("---")

st.caption(
    "🚀 Created by Vishakha Badgujar | AI Logo Detection Project"
)
