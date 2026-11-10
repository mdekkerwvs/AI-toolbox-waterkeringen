"""Dijkinspectie 0.2 — Streamlit viewer and model integration point.

This application does not claim to detect cracks until a verified segmentation
model is installed and a compatible model_adapter.py is provided.
"""
import io
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import streamlit as st

st.set_page_config(page_title='Dijkinspectie | AI Toolbox Waterkeringen', page_icon='🛡️', layout='wide')
st.title('Dijkinspectie')
st.caption('Versie 0.2 · Proefomgeving voor de AI Toolbox Waterkeringen')
st.warning('Geen gevalideerde scheurdetectie aangesloten. Gemarkeerde resultaten zijn alleen beschikbaar met een aangeleverd detectiemasker of een gekoppeld model.')

with st.expander('Over deze toepassing en de beperkingen'):
    st.write('Deze interface toont dijkbeelden, kan een bestaand pixelmasker visualiseren en ondersteunt een toekomstige koppeling met de AI Toolbox. Een masker is een zwart-witbeeld waarin witte pixels mogelijke scheuren voorstellen.')
    st.write('Bij echt gebruik moeten resultaten door een waterkeringinspecteur worden gecontroleerd. Deze app stelt geen veiligheidsoordeel vast. Gebruik hier alleen beelden die geschikt zijn voor verwerking via een publieke clouddienst.')

photo_file = st.file_uploader('Upload een dijkfoto (JPG/PNG)', type=['jpg','jpeg','png'], key='photo')
if photo_file is None:
    st.info('Upload een foto om te beginnen.')
    st.stop()

try:
    photo = ImageOps.exif_transpose(Image.open(photo_file)).convert('RGB')
except Exception:
    st.error('De afbeelding kon niet worden geopend.')
    st.stop()

MAX_SIDE = 2400
if max(photo.size) > MAX_SIDE:
    ratio = MAX_SIDE / max(photo.size)
    photo = photo.resize((max(1,round(photo.width*ratio)),max(1,round(photo.height*ratio))),Image.Resampling.LANCZOS)
    st.info('Grote foto verkleind voor weergave (maximaal 2400 pixels per zijde).')

st.image(photo, caption='Originele dijkfoto', use_container_width=True)

adapter_exists = Path('model_adapter.py').is_file()
method_options = ['Bestaand detectiemasker uploaden']
if adapter_exists:
    method_options.insert(0,'AI-model uitvoeren (experimenteel)')
method = st.radio('Analysebron',method_options)
mask = None
if method == 'Bestaand detectiemasker uploaden':
    mask_file = st.file_uploader('Upload bijbehorend zwart-wit detectiemasker (PNG)',type=['png'],key='mask')
    if mask_file:
        try:
            mask_image = Image.open(mask_file).convert('L')
            if mask_image.size != photo.size:
                st.warning('Het masker heeft een andere resolutie. Het wordt aangepast aan het formaat van de foto; controleer de uitlijning.')
                mask_image = mask_image.resize(photo.size,Image.Resampling.NEAREST)
            threshold=st.slider('Drempel voor detectiemasker',0,255,127)
            mask = np.asarray(mask_image) > threshold
        except Exception:
            st.error('Het masker kon niet worden gelezen.')
else:
    st.caption('Het gekoppelde model is experimenteel en is niet gevalideerd voor waterkeringinspecties.')
    if st.button('Analyseer met gekoppeld model',type='primary'):
        try:
            from model_adapter import predict_mask
            predicted = np.asarray(predict_mask(photo))
            if predicted.shape != (photo.height, photo.width):
                raise ValueError('Modelresultaat moet een 2D-masker zijn met dezelfde hoogte en breedte als de afbeelding.')
            mask = predicted.astype(bool) if predicted.dtype == np.bool_ else (predicted.astype('float32') >= 0.5)
        except Exception as exc:
            st.error(f'Model kon niet worden uitgevoerd: {exc}')

if mask is not None:
    alpha = st.slider('Transparantie van markering',0.1,0.9,0.55,0.05)
    base = np.asarray(photo).copy()
    overlay = base.astype('float32')
    overlay[mask] = (1-alpha)*overlay[mask] + alpha*np.array([255,0,0],dtype='float32')
    marked = Image.fromarray(np.uint8(np.clip(overlay,0,255)))
    left,right=st.columns(2)
    with left:
        st.image(photo,caption='Origineel',use_container_width=True)
    with right:
        st.image(marked,caption='Markering uit aangeleverd masker of gekoppeld model',use_container_width=True)
    percentage = 100*float(mask.mean())
    st.metric('Gemarkeerde beeldpixels',f'{percentage:.2f}%')
    st.caption('Dit percentage zegt niets over de ernst, lengte of breedte van scheuren. Geen geotechnische beoordeling.')
    output=io.BytesIO()
    marked.save(output,format='PNG')
    st.download_button('Download gemarkeerde afbeelding',data=output.getvalue(),file_name='dijkinspectie_markering.png',mime='image/png')
else:
    st.info('Nog geen detectieresultaat. Upload een bijpassend masker. Voor automatische detectie is een werkende model_adapter.py met getrainde modelgewichten nodig.')

st.divider()
st.caption('Proefapplicatie · Geen vervanging van inspecties door deskundigen · Bron: AI Toolbox Waterkeringen (HHNK / STOWA / RWS-WVL / HKV)')
