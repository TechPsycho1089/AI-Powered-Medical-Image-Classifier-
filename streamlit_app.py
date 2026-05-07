"""
streamlit_app.py  —  Malaria Cell Classifier Web App
======================================================
Run with:
    pip install streamlit requests pillow
    streamlit run streamlit_app.py

Make sure your Flask server is running first:
    python3 run_app.py
"""

import streamlit as st
import requests
import base64
from PIL import Image
import io

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Malaria Cell Classifier",
    page_icon="🔬",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { max-width: 720px; margin: 0 auto; }
    .stApp { background-color: #0f1117; }
    h1 { font-size: 1.6rem !important; font-weight: 600 !important; }
    .result-infected {
        background: #2d1515;
        border: 1px solid #c0392b;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
    }
    .result-healthy {
        background: #0f2a1a;
        border: 1px solid #27ae60;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin: 1rem 0;
    }
    .metric-row {
        display: flex;
        gap: 12px;
        margin-top: 0.8rem;
    }
    .metric-box {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        padding: 0.6rem 1rem;
        flex: 1;
        text-align: center;
    }
    .metric-label { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 22px; font-weight: 600; margin-top: 2px; }
    .disclaimer {
        font-size: 11px;
        color: #555;
        text-align: center;
        margin-top: 2rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown("## 🔬 Malaria Cell Classifier")
st.markdown("*Biocon Ltd. · UCS321 Project #4 · CNN + Grad-CAM*")
st.divider()

# ── Server config ─────────────────────────────────────────────
API_URL = "http://localhost:8080/predict"

# Check server status
@st.cache_data(ttl=10)
def check_server():
    try:
        r = requests.get("http://localhost:8080/health", timeout=2)
        return r.json()
    except:
        return None

status = check_server()
if status and status.get("model_loaded"):
    st.success("✅ Model server is running and ready", icon=None)
elif status:
    st.warning("⚠️ Server running but model not loaded (demo mode)")
else:
    st.error("❌ Server not running. Start it with: `python3 run_app.py`")
    st.stop()

st.markdown("---")

# ── Upload ────────────────────────────────────────────────────
st.markdown("### Upload a cell image")
uploaded = st.file_uploader(
    "Drag and drop or click to browse",
    type=["png", "jpg", "jpeg", "bmp", "tiff"],
    label_visibility="collapsed"
)

if uploaded:
    # Show original image
    col1, col2 = st.columns(2)
    pil_img = Image.open(uploaded)

    with col1:
        st.markdown("**Original image**")
        st.image(pil_img, width=300)

    # ── Send to Flask API ──────────────────────────────────────
    with st.spinner("Analyzing cell..."):
        try:
            uploaded.seek(0)
            response = requests.post(
                API_URL,
                files={"file": (uploaded.name, uploaded.read(), uploaded.type)},
                timeout=30
            )
            result = response.json()

            if "error" in result:
                st.error(f"Error: {result['error']}")
                st.stop()

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to server. Make sure `python3 run_app.py` is running.")
            st.stop()
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.stop()

    # ── Show Grad-CAM ──────────────────────────────────────────
    if result.get("gradcam_b64"):
        with col2:
            st.markdown("**Grad-CAM overlay**")
            gradcam_bytes = base64.b64decode(result["gradcam_b64"])
            gradcam_img = Image.open(io.BytesIO(gradcam_bytes))
            st.image(gradcam_img, width=300)

    # ── Result ─────────────────────────────────────────────────
    label      = result["label"]
    confidence = result["confidence"]
    p_inf      = result["probability_infected"]
    p_hlt      = result["probability_healthy"]
    risk       = result.get("risk_level", "")
    demo       = result.get("demo_mode", False)

    if demo:
        st.warning("⚠️ Running in demo mode — predictions are random. Train model first.")

    st.markdown("---")
    st.markdown("### Result")

    if label == "Infected":
        st.markdown(f"""
        <div class="result-infected">
            <div style="font-size:1.3rem; font-weight:600; color:#e74c3c;">
                🦠 Malaria Infected
            </div>
            <div style="color:#aaa; font-size:13px; margin-top:4px;">
                Risk Level: <strong style="color:#e74c3c">{risk}</strong>
            </div>
            <div class="metric-row">
                <div class="metric-box">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value" style="color:#e74c3c">{confidence*100:.1f}%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">P(Infected)</div>
                    <div class="metric-value" style="color:#e74c3c">{p_inf*100:.1f}%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">P(Healthy)</div>
                    <div class="metric-value" style="color:#888">{p_hlt*100:.1f}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-healthy">
            <div style="font-size:1.3rem; font-weight:600; color:#2ecc71;">
                ✅ Healthy Cell
            </div>
            <div style="color:#aaa; font-size:13px; margin-top:4px;">
                Risk Level: <strong style="color:#2ecc71">{risk}</strong>
            </div>
            <div class="metric-row">
                <div class="metric-box">
                    <div class="metric-label">Confidence</div>
                    <div class="metric-value" style="color:#2ecc71">{confidence*100:.1f}%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">P(Healthy)</div>
                    <div class="metric-value" style="color:#2ecc71">{p_hlt*100:.1f}%</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">P(Infected)</div>
                    <div class="metric-value" style="color:#888">{p_inf*100:.1f}%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Confidence bar ─────────────────────────────────────────
    st.markdown("**Prediction confidence**")
    col_h, col_i = st.columns(2)
    with col_h:
        st.markdown("Healthy")
        st.progress(float(p_hlt))
        st.caption(f"{p_hlt*100:.2f}%")
    with col_i:
        st.markdown("Infected")
        st.progress(float(p_inf))
        st.caption(f"{p_inf*100:.2f}%")

    # ── Preprocessed image ─────────────────────────────────────
    if result.get("preprocessed_b64"):
        with st.expander("View preprocessed image (CLAHE normalized)"):
            pre_bytes = base64.b64decode(result["preprocessed_b64"])
            pre_img = Image.open(io.BytesIO(pre_bytes))
            st.image(pre_img, width=200)

st.markdown("""
<div class="disclaimer">
    For educational and research purposes only · Not a medical device<br>
    Always consult qualified medical professionals for diagnosis
</div>
""", unsafe_allow_html=True)

# lsof -ti:8080 | xargs kill -9
# python3 run_app.py

# streamlit run streamlit_app.py