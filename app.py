"""Dijkinspectie demo using the original STOWA/HHNK TensorFlow SavedModel.
Not a validated safety inspection instrument.
"""
import io
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st

st.set_page_config(page_title="Dijkinspectie – STOWA-model", layout="wide")
st.title("Dijkinspectie – STOWA scheurdetectie")
st.warning("Demonstratie. AI-uitkomsten zijn niet gevalideerd voor veiligheidsbesluiten; een deskundige moet de detecties controleren.")

MODEL_DIR = (Path(__file__).resolve().parent / "scripts" / "scheur-detectie" / "saved_weights" / "model_v2_plateau" / "BATCHSIZE_80_LR_0.0001_EPOCH_150_Fold_4")

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
    return tf.keras.models.load_model(str(MODEL_DIR), custom_objects={"dice_loss": dice_loss, "dice_coef": dice_coef}, compile=False)

def predict_patches(model, rgb, progress):
    h, w = rgb.shape[:2]
    # Pad to multiples of 256, preserving every pixel in the source image.
    H = ((h + 255) // 256) * 256
    W = ((w + 255) // 256) * 256
    if h == 0 or w == 0:
        raise ValueError("Afbeelding is leeg")
    pad = np.pad(rgb, ((0, H-h), (0, W-w), (0, 0)), mode="edge")
    heat = np.zeros((H, W), dtype=np.float32)
    total = (H//256) * (W//256)
    count = 0
    for y in range(0, H, 256):
        for x in range(0, W, 256):
            patch = pad[y:y+256, x:x+256].astype(np.float32) / 255.0
            pred = model.predict(patch[None, ...], verbose=0)
            if isinstance(pred, (list, tuple)):
                pred = pred[0]
            pred = np.asarray(pred)
            if pred.shape == (1, 256, 256, 1):
                tile = pred[0, :, :, 0]
            elif pred.shape == (1, 256, 256):
                tile = pred[0]
            else:
                raise ValueError(f"Onverwachte modeluitvoer: {pred.shape}. Verwacht (1,256,256,1).")
            heat[y:y+256, x:x+256] = tile
            count += 1
            progress.progress(count/total)
    return heat[:h, :w]

st.caption("Bron: AI Toolbox Waterkeringen (HHNK / HKV / STOWA). Origineel U-Net SavedModel uit de GitHub-repository.")
if not (MODEL_DIR / "saved_model.pb").is_file() or not (MODEL_DIR / "variables").is_dir():
    st.error("Modelbestanden ontbreken. Controleer of de volledige map 'scripts/scheur-detectie/saved_weights' in je GitHub-fork staat.")
    st.stop()

file = st.file_uploader("Upload een dijkfoto (JPG of PNG)", type=["jpg", "jpeg", "png"])
threshold = st.slider("Detectiedrempel (modelscore)", min_value=0.05, max_value=0.95, value=0.50, step=0.05)
st.caption("De drempel is een modelscore, geen bewezen kans op een echte scheur. RGB-foto's zijn mogelijk minder geschikt dan de oorspronkelijke drone-orthofoto's.")

if file is not None:
    try:
        img = Image.open(file).convert("RGB")
        w, h = img.size
        st.image(img, caption=f"Originele foto – {w} × {h} pixels", use_container_width=True)
        tiles = ((w+255)//256) * ((h+255)//256)
        st.info(f"De analyse verwerkt {tiles} beeldblokken van 256×256 pixels. Gebruik voor deze demo liefst een uitsnede van circa 1024×1024 pixels.")
        if st.button("Analyseer scheuren met STOWA-model", type="primary"):
            with st.spinner("Model laden en foto analyseren…"):
                model = load_stowa_model()
                progress = st.progress(0)
                rgb = np.asarray(img, dtype=np.uint8)
                scores = predict_patches(model, rgb, progress)
                mask = scores >= threshold
                overlay = rgb.astype(np.float32).copy()
                overlay[mask] = 0.45 * overlay[mask] + 0.55 * np.array([255, 0, 0], dtype=np.float32)
                overlay = np.clip(overlay, 0, 255).astype(np.uint8)
                result = Image.fromarray(overlay)
                a, b = st.columns(2)
                with a:
                    st.image(img, caption="Origineel", use_container_width=True)
                with b:
                    st.image(result, caption="Door AI gemarkeerde mogelijke scheuren", use_container_width=True)
                output = io.BytesIO()
                result.save(output, format="PNG")
                st.download_button("Download gemarkeerde foto", data=output.getvalue(), file_name="scheurdetectie_stowa.png", mime="image/png")
                st.caption(f"Gemarkeerde pixels: {100*np.mean(mask):.2f}% van het beeld. Dit is geen inspectiekwaliteitsscore.")
    except Exception as exc:
        st.error("Analyse kon niet worden uitgevoerd. Bekijk de technische fout hieronder en controleer zo nodig de Streamlit-logs.")
        st.exception(exc)

with st.expander("Technische informatie"):
    st.code(str(MODEL_DIR))
    st.write("Deze demo gebruikt TensorFlow/Keras 2.15 voor het oudere SavedModel-formaat. Streamlit Community Cloud heeft beperkte RAM; bij geheugenproblemen is een grotere server nodig.")
    st.markdown("[Origineel scheurdetectievoorbeeld](https://aitoolboxwaterkeringen.nl/examples/scheurdetectie_voorbeeld.html)")
