"""Dijkinspectie demo: STOWA/HHNK U-Net met inline beeldweergave en download.

De resultaten zijn niet gevalideerd voor veiligheidsbesluiten.
"""
import base64
import hashlib
import html
import io
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st

st.set_page_config(page_title="Dijkinspectie – STOWA-model", layout="wide")
st.title("Dijkinspectie – STOWA scheurdetectie")
st.warning("Demonstratie. AI-uitkomsten zijn niet gevalideerd voor veiligheidsbesluiten; een deskundige moet de detecties controleren.")

MODEL_DIR = (
    Path(__file__).resolve().parent
    / "scripts" / "scheur-detectie" / "saved_weights" / "model_v2_plateau"
    / "BATCHSIZE_80_LR_0.0001_EPOCH_150_Fold_4"
)

@st.cache_resource(show_spinner="STOWA U-Net-model laden…")
def load_stowa_model():
    import tensorflow as tf

    def dice_coef(y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true * y_pred)
        smooth = tf.constant(1e-15, dtype=tf.float32)
        return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

    def dice_loss(y_true, y_pred):
        return 1.0 - dice_coef(y_true, y_pred)

    return tf.keras.models.load_model(
        str(MODEL_DIR),
        custom_objects={"dice_loss": dice_loss, "dice_coef": dice_coef},
        compile=False,
    )


def predict_patches(model, rgb, progress):
    h, w = rgb.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Afbeelding is leeg")
    H = ((h + 255) // 256) * 256
    W = ((w + 255) // 256) * 256
    padded = np.pad(rgb, ((0, H - h), (0, W - w), (0, 0)), mode="edge")
    scores = np.zeros((H, W), dtype=np.float32)
    total = (H // 256) * (W // 256)
    count = 0
    for y in range(0, H, 256):
        for x in range(0, W, 256):
            patch = padded[y:y + 256, x:x + 256].astype(np.float32) / 255.0
            prediction = model.predict(patch[None, ...], verbose=0)
            if isinstance(prediction, (list, tuple)):
                prediction = prediction[0]
            prediction = np.asarray(prediction)
            if prediction.shape == (1, 256, 256, 1):
                tile = prediction[0, :, :, 0]
            elif prediction.shape == (1, 256, 256):
                tile = prediction[0]
            else:
                raise ValueError(f"Onverwachte modeluitvoer: {prediction.shape}")
            scores[y:y + 256, x:x + 256] = tile
            count += 1
            progress.progress(count / total)
    return scores[:h, :w]


def png_bytes(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def embedded_image(png, caption):
    """Embed via data URI: geen Streamlit /media-bestandsverzoek nodig."""
    data = base64.b64encode(png).decode("ascii")
    caption = html.escape(caption)
    st.markdown(
        f'<figure style="margin:0"><img src="data:image/png;base64,{data}" '
        f'style="width:100%;height:auto;border-radius:9px" alt="{caption}">'
        f'<figcaption style="text-align:center;color:gray;margin-top:8px">{caption}</figcaption></figure>',
        unsafe_allow_html=True,
    )


def direct_download(png, filename):
    """Download directly embedded data, without Streamlit media endpoint."""
    data = base64.b64encode(png).decode("ascii")
    st.markdown(
        f'<a download="{filename}" href="data:image/png;base64,{data}" '
        'style="display:inline-block;border:1px solid #999;border-radius:8px;'
        'padding:10px 16px;text-decoration:none">'
        '⬇ Download gemarkeerde foto (alternatief)</a>',
        unsafe_allow_html=True,
    )


st.caption("Bron: AI Toolbox Waterkeringen (HHNK / HKV / STOWA). Origineel U-Net SavedModel uit de GitHub-repository.")
if not (MODEL_DIR / "saved_model.pb").is_file() or not (MODEL_DIR / "variables").is_dir():
    st.error("Modelbestanden ontbreken. Controleer de map scripts/scheur-detectie/saved_weights in GitHub.")
    st.stop()

uploaded = st.file_uploader("Upload een dijkfoto (JPG of PNG)", type=["jpg", "jpeg", "png"])
threshold = st.slider("Detectiedrempel (modelscore)", 0.05, 0.95, 0.50, 0.05)
st.caption("De drempel is een modelscore, geen bewezen kans op een echte scheur.")

if uploaded is not None:
    try:
        raw = uploaded.getvalue()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        w, h = img.size
        img_png = png_bytes(img)
        embedded_image(img_png, f"Originele foto – {w} × {h} pixels")
        tiles = ((w + 255) // 256) * ((h + 255) // 256)
        st.info(f"De analyse verwerkt {tiles} beeldblokken van 256×256 pixels.")

        # Invalidate previously analysed image when photo or threshold changes.
        fingerprint = (hashlib.sha256(raw).hexdigest(), float(threshold))
        if st.session_state.get("analysed_fingerprint") != fingerprint:
            st.session_state.pop("analysed_png", None)
            st.session_state.pop("analysed_percentage", None)

        if st.button("Analyseer scheuren met STOWA-model", type="primary"):
            with st.spinner("STOWA-model laden en foto analyseren…"):
                model = load_stowa_model()
                progress = st.progress(0)
                rgb = np.asarray(img, dtype=np.uint8)
                scores = predict_patches(model, rgb, progress)
                mask = scores >= threshold
                overlay = rgb.astype(np.float32).copy()
                overlay[mask] = 0.45 * overlay[mask] + 0.55 * np.array([255, 0, 0], dtype=np.float32)
                overlay = np.clip(overlay, 0, 255).astype(np.uint8)
                result = Image.fromarray(overlay, mode="RGB")
                st.session_state["analysed_png"] = png_bytes(result)
                st.session_state["analysed_percentage"] = 100 * float(np.mean(mask))
                st.session_state["analysed_fingerprint"] = fingerprint

        if st.session_state.get("analysed_fingerprint") == fingerprint and "analysed_png" in st.session_state:
            result_png = st.session_state["analysed_png"]
            a, b = st.columns(2)
            with a:
                embedded_image(img_png, "Origineel")
            with b:
                embedded_image(result_png, "Door AI gemarkeerde mogelijke scheuren")
            st.download_button(
                "Download gemarkeerde foto (Streamlit)",
                data=result_png,
                file_name="scheurdetectie_stowa.png",
                mime="image/png",
            )
            direct_download(result_png, "scheurdetectie_stowa.png")
            st.caption(
                f"Gemarkeerde pixels: {st.session_state['analysed_percentage']:.2f}% van het beeld. "
                "Dit is geen inspectiekwaliteitsscore."
            )
            st.caption("Werkt de gewone download niet? Probeer de alternatieve knop. "
                       "Je kunt ook met de rechtermuisknop op de gemarkeerde afbeelding klikken en 'Afbeelding opslaan als' kiezen.")
    except Exception as exc:
        st.error("De analyse kon niet worden uitgevoerd.")
        st.exception(exc)

with st.expander("Technische informatie"):
    st.code(str(MODEL_DIR))
    st.write("TensorFlow/Keras 2.15, Python 3.11. Inline PNG-weergave om problemen met Streamlit-media-URLs te vermijden.")
    st.markdown("[Origineel scheurdetectievoorbeeld](https://aitoolboxwaterkeringen.nl/examples/scheurdetectie_voorbeeld.html)")
