"""Dijkinspectie — Streamlit-frontend voor AI Toolbox Waterkeringen.

Deze basisapp bevat GEEN scheurdetectiemodel. Sluit de gevalideerde
inferentiecode en modelgewichten uit de AI Toolbox afzonderlijk aan.
"""
from io import BytesIO
from PIL import Image, ImageOps, UnidentifiedImageError
import streamlit as st

MAX_IMAGE_BYTES = 15 * 1024 * 1024

st.set_page_config(page_title="Dijkinspectie | Demonstratie", page_icon="🛡️", layout="wide")
st.title("Dijkinspectie")
st.caption("Demonstratieomgeving — voorbereiding op AI Toolbox Waterkeringen")
st.warning(
    "AI-scheurdetectie is nog niet aangesloten. Deze versie toont afbeeldingen, "
    "maar voert geen automatische inspectie uit."
)

with st.expander("Over deze toepassing", expanded=False):
    st.markdown(
        "Deze proefinterface is bedoeld om het inspectieproces uit te testen. "
        "Het daadwerkelijke detectiemodel, de modelgewichten en de juiste "
        "voorverwerking moeten nog worden aangesloten. Resultaten zijn geen "
        "vervanging voor een beoordeling door een bevoegde inspecteur.\n\n"
        "Bronproject: [AI Toolbox Waterkeringen](https://github.com/AI-Toolbox-Waterkeringen/AI-toolbox-waterkeringen)."
    )

uploaded = st.file_uploader("Upload een foto van een dijk", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    if uploaded.size > MAX_IMAGE_BYTES:
        st.error("Bestand te groot. Gebruik een afbeelding van maximaal 15 MB.")
        st.stop()
    try:
        image = Image.open(BytesIO(uploaded.getvalue()))
        image.verify()
        image = Image.open(BytesIO(uploaded.getvalue()))
        image = ImageOps.exif_transpose(image).convert("RGB")
        if image.width * image.height > 40_000_000:
            st.error("Afbeelding heeft te veel pixels. Verklein de foto eerst.")
            st.stop()
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("Dit bestand is geen geldige afbeelding.")
        st.stop()

    left, right = st.columns(2)
    with left:
        st.subheader("Originele foto")
        st.image(image, use_container_width=True)
    with right:
        st.subheader("AI-resultaat")
        st.info("Nog niet beschikbaar: er is geen detectiemodel gekoppeld.")
        st.button("Analyseer op scheuren", disabled=True, help="Beschikbaar na koppeling van de AI Toolbox-inferentie.")
    st.caption(f"Foto: {image.width} × {image.height} pixels. Afbeelding wordt niet bewust permanent opgeslagen door deze app.")
else:
    st.info("Kies hierboven een JPG- of PNG-afbeelding om de interface te testen.")

st.divider()
st.caption("Pilotinterface • Geen veiligheidsbeoordeling • Gebruik uitsluitend niet-vertrouwelijke testbeelden op publieke hosting.")
