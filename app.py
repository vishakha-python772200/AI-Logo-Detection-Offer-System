# ==========================================
# IMPORT LIBRARIES
# ==========================================
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="AI Logo Detection System",
    page_icon="AI",
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
    text-align: center;
    color: white;
    font-size: 45px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 18px;
}
.offer-box {
    background: #ef4444;
    padding: 20px;
    border-radius: 15px;
    color: white;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
    margin-top: 15px;
}
.result-box {
    background: #1e293b;
    padding: 20px;
    border-radius: 15px;
    color: white;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# MODEL LOAD
# ==========================================
MODEL_PATH = r"best.pt"

if not os.path.exists(MODEL_PATH):
    st.error("Model nahi milala! Pahile train.py run kara!")
    st.stop()

model = YOLO(MODEL_PATH)

# ==========================================
# 10 BRANDS - OFFERS
# ==========================================
OFFERS = {
    "kfc":      "50% OFF ON ALL COMBOS TODAY",
    "mcd":      "BUY 1 BURGER GET 1 FREE",
    "vw":       "1 DAY FREE TEST DRIVE",
    "bata":     "50% OFF ON ALL FOOTWEAR",
    "bmw":      "NEW BMW MODEL LAUNCHED - BOOK NOW",
    "dominos":  "BUY 1 PIZZA GET 1 FREE",
    "fastrack": "20% OFF ON ALL WATCHES",
    "haldiram": "BUY 1 GET 1 FREE ON SWEETS",
    "starbuks": "BUY 2 COFFEE GET 1 FREE",
    "zudio":    "10% OFF ON ALL CLOTHES"
}

# ==========================================
# HEADER
# ==========================================
st.markdown(
    '<p class="title">AI Logo Detection & Offer System</p>',
    unsafe_allow_html=True
)
st.markdown(
    '<p class="subtitle">Upload Brand Logo - Get Instant Offer!</p>',
    unsafe_allow_html=True
)
st.write("")

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("Project Menu")
st.sidebar.success("AI Logo Classification System")
st.sidebar.info("""
Supported Brands:
- KFC
- McDonalds
- Volkswagen
- Bata
- BMW
- Dominos
- Fastrack
- Haldiram
- Starbucks
- Zudio
""")
st.sidebar.write("---")
st.sidebar.caption("Created by Vishakha Badgujar")

# ==========================================
# INSTRUCTIONS
# ==========================================
st.info("""
How to Use:
1. Open camera on mobile or laptop
2. Capture brand logo image
3. Upload image below
4. AI will detect logo automatically
5. Offer will be displayed instantly
""")

# ==========================================
# FILE UPLOAD
# ==========================================
uploaded_file = st.file_uploader(
    "Upload Logo Image",
    type=["jpg", "jpeg", "png", "webp"]
)

# ==========================================
# PREDICTION
# ==========================================
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(
            image,
            caption="Your Logo",
            use_container_width=True
        )

    # PIL Image → numpy array → YOLO
    # tempfile नाही - Cloud वर काम करतो!
    img_array = np.array(image.convert("RGB"))

    # YOLO Prediction
    results    = model.predict(
                    source  = img_array,
                    imgsz   = 224,
                    verbose = False
                )
    r          = results[0]
    pred_id    = int(r.probs.top1)
    confidence = float(r.probs.top1conf)
    pred_name  = r.names[pred_id].lower()
    offer      = OFFERS.get(pred_name, "WELCOME - SPECIAL OFFER AVAILABLE")

    with col2:
        st.subheader("Detection Result")

        # Result Box
        st.markdown(
            f"""
            <div class="result-box">
                <h2>Detected Brand</h2>
                <h1>{pred_name.upper()}</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        # Confidence Score
        st.success(f"Confidence Score : {confidence * 100:.2f}%")
        st.progress(int(confidence * 100))

        st.write("")

        # Offer Box
        st.markdown(
            f"""
            <div class="offer-box">
                SPECIAL OFFER<br>
                {offer}
            </div>
            """,
            unsafe_allow_html=True
        )

# ==========================================
# BRAND CARDS
# ==========================================
st.write("")
st.write("---")
st.subheader("Supported Brands")

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
st.caption("Created by Vishakha Badgujar | AI Logo Detection Project")
