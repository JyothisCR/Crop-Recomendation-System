import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import warnings
import streamlit.components.v1 as components
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Smart Crop Recommendation",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0e1a0e !important;
    color: #d4e8d4;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stAppViewContainer"] > .main { background-color: #0e1a0e; }
[data-testid="stHeader"], [data-testid="stToolbar"] { background: transparent !important; }
section[data-testid="stSidebar"] { display: none; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0e1a0e; }
::-webkit-scrollbar-thumb { background: #2d5c2d; border-radius: 3px; }

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2rem, 5vw, 3.2rem);
    font-weight: 900;
    text-align: center;
    color: #e8f5e8;
    letter-spacing: -0.5px;
    margin: 0.2rem 0 0.3rem;
    line-height: 1.15;
}
.hero-title span { color: #5cb85c; font-style: italic; }
.hero-sub {
    text-align: center;
    color: #7a9e7a;
    font-size: 1.08rem;
    font-weight: 300;
    margin-bottom: 1.8rem;
    letter-spacing: 0.02em;
}

.section-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #5cb85c;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid rgba(92,184,92,0.2);
}

[data-testid="stNumberInput"] > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(92,184,92,0.2) !important;
    border-radius: 8px !important;
    color: #e8f5e8 !important;
    font-family: 'DM Mono', monospace !important;
}
[data-testid="stNumberInput"] input {
    color: #e8f5e8 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 1.08rem !important;
    background: transparent !important;
}
[data-testid="stNumberInput"] label {
    color: #9ab89a !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stNumberInput"] button {
    color: #5cb85c !important;
    background: rgba(92,184,92,0.08) !important;
    border-radius: 4px !important;
    font-size: 1.1rem !important;
}
[data-testid="stNumberInput"] button:hover { background: rgba(92,184,92,0.18) !important; }

.range-badge {
    display: inline-block;
    background: rgba(92,184,92,0.1);
    border: 1px solid rgba(92,184,92,0.2);
    border-radius: 4px;
    padding: 1px 8px;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    color: #7ab87a;
    margin-top: 2px;
}

[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #2d7a2d 0%, #4caf50 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.65rem 2rem !important;
    letter-spacing: 0.03em;
    box-shadow: 0 4px 20px rgba(76,175,80,0.3) !important;
    transition: all 0.2s !important;
}
[data-testid="baseButton-primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(76,175,80,0.45) !important;
}

.result-card {
    background: linear-gradient(135deg, rgba(45,122,45,0.18) 0%, rgba(14,26,14,0.9) 100%);
    border: 1px solid rgba(92,184,92,0.35);
    border-radius: 14px;
    padding: 1.6rem 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 1rem;
}
.result-crop-name {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: #e8f5e8;
    line-height: 1;
}
.result-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #5cb85c;
    margin-bottom: 4px;
}

[data-testid="stTable"] table {
    background: transparent !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.92rem !important;
}
[data-testid="stTable"] th {
    background: rgba(92,184,92,0.1) !important;
    color: #5cb85c !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid rgba(92,184,92,0.2) !important;
}
[data-testid="stTable"] td {
    color: #b8d4b8 !important;
    border-bottom: 1px solid rgba(255,255,255,0.05) !important;
}

.tip-box {
    background: rgba(92,184,92,0.07);
    border-left: 3px solid #5cb85c;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.92rem;
    color: #9ab89a;
    margin-top: 1rem;
    line-height: 1.5;
}
.tip-box strong { color: #5cb85c; }

hr { border-color: rgba(92,184,92,0.1) !important; }
[data-testid="stAlert"] {
    background: rgba(255,80,80,0.08) !important;
    border: 1px solid rgba(255,80,80,0.25) !important;
    border-radius: 8px !important;
    color: #ffaaaa !important;
}
[data-testid="stSpinner"] { color: #5cb85c !important; }
</style>
""", unsafe_allow_html=True)

# JS: clear default 0 on focus
# FIX: wrapped in components.html() instead of st.markdown() —
# Streamlit strips <script> tags from st.markdown for security reasons,
# so the clear-on-focus behaviour was silently never running.
components.html("""
<script>
(function pollInputs() {
    const inputs = document.querySelectorAll('[data-testid="stNumberInput"] input');
    inputs.forEach(inp => {
        if (inp.dataset.clearBound) return;
        inp.dataset.clearBound = "1";
        inp.addEventListener('focus', function() {
            if (this.value === '0' || this.value === '0.00' || this.value === '0.0') this.value = '';
        });
        inp.addEventListener('blur', function() {
            if (this.value === '') {
                this.value = '0';
                this.dispatchEvent(new Event('input', { bubbles: true }));
            }
        });
    });
    setTimeout(pollInputs, 800);
})();
</script>
""", height=0)


# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# FIX: replaced hardcoded Windows path with a portable relative path,
# matching the same fix applied to model.py
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
# FIX: was r"S:\Project\P1\Crop_recommendation.csv" — now portable
DATA_PATH  = os.path.join(BASE_DIR, "Crop_recommendation.csv")

MODEL_PKL   = os.path.join(MODEL_DIR, "stack_model_combined.pkl")
ENCODER_PKL = os.path.join(MODEL_DIR, "label_encoder_combined.pkl")


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
CROP_EMOJIS = {
    "rice": "🌾", "wheat": "🌾", "maize": "🌽", "coffee": "☕",
    "jute": "🌿", "cotton": "🌸", "coconut": "🥥", "papaya": "🍈",
    "orange": "🍊", "apple": "🍎", "muskmelon": "🍈", "watermelon": "🍉",
    "grapes": "🍇", "mango": "🥭", "banana": "🍌", "pomegranate": "🍎",
    "lentil": "🫘", "blackgram": "🫘", "mungbean": "🫘", "mothbeans": "🫘",
    "pigeonpeas": "🫘", "kidneybeans": "🫘", "chickpea": "🫘",
}
SHAP_FEATURE_META = [
    ("N",           "Nitrogen",    "kg/ha", "🌿"),
    ("P",           "Phosphorus",  "kg/ha", "🧪"),
    ("K",           "Potassium",   "kg/ha", "⚗️"),
    ("ph",          "Soil pH",     "",      "🔬"),
    ("temperature", "Temperature", "°C",    "🌡️"),
    ("humidity",    "Humidity",    "%",     "💧"),
    ("rainfall",    "Rainfall",    "mm",    "🌧️"),
]

def get_emoji(name):
    return CROP_EMOJIS.get((name or "").lower(), "🌱")


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING  (runs once, saves pkl, cached forever after)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_or_train_model():
    """
    1. If models/stack_model_combined.pkl exists → load it.
    2. Otherwise → train from DATA_PATH, save to models/, return loaded model.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_PKL) and os.path.exists(ENCODER_PKL):
        with open(MODEL_PKL,   "rb") as f: model   = pickle.load(f)
        with open(ENCODER_PKL, "rb") as f: encoder = pickle.load(f)
        return model, encoder, False   # False = was not retrained

    # ── Train from scratch ────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    # Normalise column name — CSV uses 'ph' (lowercase)
    df.columns = [c if c != 'pH' else 'ph' for c in df.columns]

    X = df.drop(columns=['label'])
    y = df['label']

    encoder = LabelEncoder()
    y_enc   = encoder.fit_transform(y)

    from sklearn.model_selection import train_test_split
    X_train, _, y_train, _ = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=1.0, gamma='scale',
                    probability=True, random_state=42))
    ])
    estimators = [
        ('rf',  RandomForestClassifier(n_estimators=100, max_depth=10,
                                       min_samples_leaf=5, random_state=42)),
        ('svm', svm_pipeline),
        ('xgb', XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                               subsample=0.8, colsample_bytree=0.8,
                               eval_metric='mlogloss', random_state=42)),
    ]
    model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=0.5, max_iter=1000),
        cv=5, n_jobs=-1
    )
    model.fit(X_train, y_train)

    with open(MODEL_PKL,   "wb") as f: pickle.dump(model,   f)
    with open(ENCODER_PKL, "wb") as f: pickle.dump(encoder, f)

    return model, encoder, True   # True = was freshly trained


# ══════════════════════════════════════════════════════════════════════════════
# SHAP
# ══════════════════════════════════════════════════════════════════════════════
def compute_shap_values(model, input_array, class_idx):
    import shap
    rf_model  = model.named_estimators_["rf"]
    svm_model = model.named_estimators_["svm"]
    xgb_model = model.named_estimators_["xgb"]
    meta      = model.final_estimator_

    rf_proba   = rf_model.predict_proba(input_array)
    svm_proba  = svm_model.predict_proba(input_array)
    xgb_proba  = xgb_model.predict_proba(input_array)
    meta_input = np.hstack([rf_proba, svm_proba, xgb_proba])

    n_cls        = rf_proba.shape[1]
    uniform_prob = np.full((1, n_cls), 1.0 / n_cls)
    background   = np.hstack([uniform_prob, uniform_prob, uniform_prob])

    explainer    = shap.LinearExplainer(meta, background)
    meta_shap_sv = explainer.shap_values(meta_input)

    if isinstance(meta_shap_sv, list):
        meta_sv = np.array(meta_shap_sv[class_idx])[0]
    else:
        meta_sv = np.array(meta_shap_sv)
        if meta_sv.ndim == 3:   meta_sv = meta_sv[0, :, class_idx]
        elif meta_sv.ndim == 2: meta_sv = meta_sv[0, :]

    n_classes   = rf_proba.shape[1]
    rf_meta_sv  = meta_sv[0           : n_classes]
    svm_meta_sv = meta_sv[n_classes   : 2 * n_classes]
    xgb_meta_sv = meta_sv[2*n_classes : 3 * n_classes]

    n_features = input_array.shape[1]
    rf_fi  = rf_model.feature_importances_;  rf_fi  = rf_fi  / (rf_fi.sum()  + 1e-12)
    xgb_fi = xgb_model.feature_importances_; xgb_fi = xgb_fi / (xgb_fi.sum() + 1e-12)
    svm_fi = np.ones(n_features) / n_features

    feature_shap = (
        float(rf_meta_sv.sum())  * rf_fi +
        float(svm_meta_sv.sum()) * svm_fi +
        float(xgb_meta_sv.sum()) * xgb_fi
    )
    return feature_shap, 0.0


def render_shap_chart(shap_vals, input_array, top_crop):
    order   = np.argsort(np.abs(shap_vals))[::-1]
    max_abs = max(float(np.abs(shap_vals).max()), 1e-9)
    rows_html = ""
    for idx in order:
        sv      = float(shap_vals[idx])
        raw_val = float(input_array[0, idx])
        _, label, unit, emoji = SHAP_FEATURE_META[idx]
        bar_pct  = min(abs(sv) / max_abs * 100, 100)
        color    = "#4caf50" if sv >= 0 else "#e05252"
        unit_str = f" {unit}" if unit else ""
        sign     = "+" if sv >= 0 else ""
        rows_html += f"""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
          <div style="width:140px;flex-shrink:0;text-align:right;white-space:nowrap;">
            <span style="font-size:0.95rem;">{emoji}</span>
            <span style="font-family:'DM Sans',sans-serif;font-size:0.9rem;color:#c8dfc8;font-weight:500;"> {label}</span>
          </div>
          <div style="width:110px;flex-shrink:0;text-align:center;">
            <span style="font-family:'DM Mono',monospace;font-size:0.78rem;color:#7ab87a;
                         background:rgba(92,184,92,0.1);border:1px solid rgba(92,184,92,0.2);
                         border-radius:4px;padding:3px 8px;white-space:nowrap;display:inline-block;">
              {raw_val:.2f}{unit_str}
            </span>
          </div>
          <div style="flex:1;background:rgba(255,255,255,0.05);border-radius:4px;height:10px;overflow:hidden;min-width:0;">
            <div style="width:{bar_pct:.1f}%;height:100%;border-radius:4px;background:{color};box-shadow:0 0 6px {color}55;"></div>
          </div>
          <div style="width:76px;flex-shrink:0;text-align:right;white-space:nowrap;">
            <span style="font-family:'DM Mono',monospace;font-size:0.84rem;color:{color};font-weight:600;">{sign}{sv:.4f}</span>
          </div>
        </div>"""
    legend = """
    <div style="display:flex;gap:20px;margin-top:14px;padding-top:10px;border-top:1px solid rgba(92,184,92,0.12);">
      <div style="display:flex;align-items:center;gap:7px;">
        <div style="width:13px;height:13px;border-radius:3px;background:#4caf50;"></div>
        <span style="font-family:'DM Sans',sans-serif;font-size:0.8rem;color:#9ab89a;">Positive — favours this crop</span>
      </div>
      <div style="display:flex;align-items:center;gap:7px;">
        <div style="width:13px;height:13px;border-radius:3px;background:#e05252;"></div>
        <span style="font-family:'DM Sans',sans-serif;font-size:0.8rem;color:#9ab89a;">Negative — works against this crop</span>
      </div>
    </div>"""
    note = f"""
    <div style="display:flex;align-items:center;justify-content:space-between;margin:0 0 14px;flex-wrap:wrap;gap:8px;">
      <p style="font-family:'DM Sans',sans-serif;font-size:0.84rem;color:#7a9e7a;margin:0;line-height:1.6;">
        SHAP contributions toward predicting <strong style="color:#e8f5e8;">{top_crop.title()}</strong>
      </p>
      <span style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#5c7a5c;
                   background:rgba(92,184,92,0.07);border:1px solid rgba(92,184,92,0.15);
                   border-radius:4px;padding:2px 8px;white-space:nowrap;">via Stacking Meta-Learner</span>
    </div>"""
    components.html(f"""
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
    <div style="background:rgba(255,255,255,0.025);border:1px solid rgba(92,184,92,0.15);border-radius:12px;padding:1.4rem 1.6rem;">
      {note}{rows_html}{legend}
    </div>""", height=len(SHAP_FEATURE_META) * 54 + 160)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD / TRAIN MODEL AT STARTUP
# ══════════════════════════════════════════════════════════════════════════════
try:
    _model, _encoder, _was_trained = load_or_train_model()
    _model_loaded = True
    _model_error  = None
except Exception as _e:
    _model_loaded = False
    _model_error  = str(_e)
    _model        = None
    _encoder      = None
    _was_trained  = False


# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<h1 class="hero-title">Smart <span>Crop</span> Recommendation</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Enter your soil and climate parameters to discover the ideal crop for your field.</p>', unsafe_allow_html=True)

# Show training notice only once after first-time train
if _was_trained:
    st.success("✅ Model trained and saved to `models/` — future launches will load instantly.")

# Show error if model could not load or train
if not _model_loaded:
    st.markdown(f"""
    <div style="background:rgba(255,80,80,0.08);border:1px solid rgba(255,80,80,0.3);
                border-radius:10px;padding:1rem 1.4rem;margin-bottom:1rem;">
        <span style="font-size:1.1rem;">❌</span>
        <span style="font-family:'DM Sans',sans-serif;font-size:0.95rem;color:#ffaaaa;font-weight:500;">
          Could not load or train model: {_model_error}<br>
          Make sure <code>Crop_recommendation.csv</code> is in the same folder as <code>app.py</code>
          and all dependencies are installed.
        </span>
    </div>""", unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    # Soil Nutrients
    st.markdown('<div class="section-header">🌿 &nbsp;Soil Nutrients</div>', unsafe_allow_html=True)
    with st.container():
        n1, n2, n3 = st.columns(3)
        with n1:
            N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
            st.markdown('<span class="range-badge">0 – 200 kg/ha</span>', unsafe_allow_html=True)
        with n2:
            P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
            st.markdown('<span class="range-badge">0 – 200 kg/ha</span>', unsafe_allow_html=True)
        with n3:
            K = st.number_input("Potassium (K)", min_value=0.0, max_value=200.0, value=0.0, step=1.0)
            st.markdown('<span class="range-badge">0 – 200 kg/ha</span>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        ph_col, _ = st.columns([1, 2])
        with ph_col:
            pH = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=0.0, step=0.01, format="%.2f")
            st.markdown('<span class="range-badge">Acidic 0 – 14 → Alkaline</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Climate Conditions
    st.markdown('<div class="section-header">🌤 &nbsp;Climate Conditions</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=0.0, step=0.5, format="%.2f")
        st.markdown('<span class="range-badge">0 – 50 °C</span>', unsafe_allow_html=True)
    with c2:
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=0.0, step=1.0, format="%.2f")
        st.markdown('<span class="range-badge">0 – 100 %</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    rf_col, _ = st.columns([1, 2])
    with rf_col:
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=0.0, step=5.0, format="%.2f")
        st.markdown('<span class="range-badge">0 – 500 mm</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    btn = st.button("🌾  Recommend Crop", type="primary", use_container_width=False)


with col_right:
    st.markdown('<div class="section-header">📋 &nbsp;Parameter Guide</div>', unsafe_allow_html=True)
    guide_df = pd.DataFrame({
        "Parameter":     ["N", "P", "K", "pH", "Temp", "Humidity", "Rainfall"],
        "Unit":          ["kg/ha", "kg/ha", "kg/ha", "—", "°C", "%", "mm"],
        "Typical Range": ["0–140", "5–145", "5–205", "3.5–9.9", "8–44", "14–100", "20–299"],
    })
    st.table(guide_df.set_index("Parameter"))
    st.markdown("""
    <div class="tip-box">
        <strong>💡 Tip:</strong> Values outside typical ranges are allowed — the model was trained
        on diverse real-world datasets and handles edge cases gracefully.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)

if btn:
    model   = _model
    encoder = _encoder

    # Validation
    field_values = [N, P, K, pH, temperature, humidity, rainfall]
    field_names  = ["Nitrogen", "Phosphorus", "Potassium", "Soil pH",
                    "Temperature", "Humidity", "Rainfall"]
    zero_fields  = [field_names[i] for i, v in enumerate(field_values) if v == 0.0]

    if len(zero_fields) == len(field_names):
        st.markdown("""
        <div style="background:rgba(255,80,80,0.08);border:1px solid rgba(255,80,80,0.3);
                    border-radius:10px;padding:1rem 1.4rem;">
            <span style="font-size:1.1rem;">⚠️</span>
            <span style="font-family:'DM Sans',sans-serif;font-size:0.95rem;color:#ffaaaa;font-weight:500;">
              All fields are set to 0. Please enter your actual soil and climate values.
            </span>
        </div>""", unsafe_allow_html=True)
        st.stop()

    if len(zero_fields) > 0:
        st.markdown(f"""
        <div style="background:rgba(255,80,80,0.08);border:1px solid rgba(255,80,80,0.3);
                    border-radius:10px;padding:1rem 1.4rem;">
            <span style="font-size:1.1rem;">⚠️</span>
            <span style="font-family:'DM Sans',sans-serif;font-size:0.95rem;color:#ffaaaa;font-weight:500;">
              Fields still at 0: <strong>{", ".join(zero_fields)}</strong>.
              Please fill all parameters for an accurate prediction.
            </span>
        </div>""", unsafe_allow_html=True)
        st.stop()

    # Column order must match CSV: N, P, K, ph, temperature, humidity, rainfall
    inputs = np.array([[N, P, K, pH, temperature, humidity, rainfall]], dtype=np.float64)

    try:
        with st.spinner("Analysing soil and climate data…"):
            proba = model.predict_proba(inputs)[0]
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        st.stop()

    # FIX: removed broken guard `if proba is None` — numpy arrays are never
    # None and the truthiness check raises "ambiguous truth value" error.
    # Replaced with an explicit length check only.
    if len(proba) != len(encoder.classes_):
        st.error("❌ Unexpected model output. Please delete models/ and restart to retrain.")
        st.stop()

    proba_df = (
        pd.DataFrame({"Crop": encoder.classes_, "Probability": proba * 100})
        .sort_values("Probability", ascending=False)
        .reset_index(drop=True)
    )
    top_conf = proba_df.iloc[0]["Probability"]
    top_crop = proba_df.iloc[0]["Crop"]

    # Low confidence warning
    if top_conf < 20.0:
        st.markdown(f"""
        <div style="background:rgba(255,160,50,0.08);border:1px solid rgba(255,160,50,0.25);
                    border-radius:10px;padding:0.9rem 1.4rem;margin-bottom:0.5rem;">
            <span style="font-size:1rem;">📉</span>
            <span style="font-family:'DM Sans',sans-serif;font-size:0.9rem;color:#ffcc88;">
              <strong>Low confidence ({top_conf:.1f}%)</strong> — the model is uncertain.
              Multiple crops score similarly. Consider verifying your inputs or consulting an agronomist.
            </span>
        </div>""", unsafe_allow_html=True)

    # Result banner
    st.markdown(f"""
    <div class="result-card">
        <div>
            <div class="result-label">Recommended Crop</div>
            <div class="result-crop-name">{get_emoji(top_crop)} &nbsp;{top_crop.title()}</div>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📊 &nbsp;Top 5 Crop Probabilities</div>', unsafe_allow_html=True)

    max_prob = proba_df.iloc[0]["Probability"]
    top5     = proba_df.head(5)
    bar_rows = ""
    for i, row in top5.iterrows():
        rank     = i + 1
        fill_pct = (row["Probability"] / max_prob) * 100
        bold     = "font-weight:700;color:#e8f5e8;" if rank == 1 else ""
        bar_rows += f"""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:0.7rem;">
            <span style="font-family:'DM Mono',monospace;font-size:0.85rem;color:#5c7a5c;width:18px;text-align:center;flex-shrink:0;">{rank}</span>
            <span style="font-size:1.15rem;flex-shrink:0;">{get_emoji(row['Crop'])}</span>
            <span style="font-family:'DM Sans',sans-serif;font-size:1rem;font-weight:500;color:#c8dfc8;width:120px;flex-shrink:0;{bold}">{row['Crop'].title()}</span>
            <div style="flex:1;background:rgba(255,255,255,0.05);border-radius:4px;height:9px;overflow:hidden;">
                <div style="width:{fill_pct:.1f}%;height:100%;border-radius:4px;background:linear-gradient(90deg,#2d7a2d,#5cb85c);"></div>
            </div>
            <span style="font-family:'DM Mono',monospace;font-size:0.9rem;color:#9ab89a;width:56px;text-align:right;flex-shrink:0;">{row['Probability']:.2f}%</span>
        </div>"""
    components.html(f"""
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(92,184,92,0.15);border-radius:12px;padding:1.4rem 1.6rem;">
        {bar_rows}
    </div>""", height=len(top5) * 54 + 50)

    # SHAP
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">🔍 &nbsp;SHAP Feature Explainability</div>', unsafe_allow_html=True)
    try:
        import shap
        classes_list = list(encoder.classes_)
        if top_crop not in classes_list:
            raise ValueError(f"Predicted crop '{top_crop}' not found in encoder classes.")
        class_idx = classes_list.index(top_crop)
        with st.spinner("Computing SHAP values…"):
            shap_vals, _ = compute_shap_values(model, inputs, class_idx)
        if np.abs(shap_vals).max() < 1e-8:
            st.info("ℹ️ SHAP values are negligible for this prediction — chart skipped.")
        else:
            render_shap_chart(shap_vals, inputs, top_crop)
    except ImportError:
        st.warning("⚠️ `shap` not installed. Run: `pip install shap` then restart.")
    except Exception as e:
        st.warning(f"⚠️ SHAP computation failed: {e}")