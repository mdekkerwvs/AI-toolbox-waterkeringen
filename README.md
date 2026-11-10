# Dijkinspectie – Streamlit proefinterface

Deze versie biedt foto-upload en weergave; **de echte AI-scheurdetectie is nog niet aangesloten**.

## Online publiceren

1. Maak een GitHub-repository (of fork de AI Toolbox Waterkeringen).
2. Upload `app.py` en `requirements.txt` naar de hoofdmap van die repository.
3. Open https://share.streamlit.io, kies **Create app**, selecteer repository, branch `main`, en bestand `app.py`.
4. Publiceer de app. Voor een privacygevoelige toepassing is een publieke gratis demo niet geschikt.

## Lokaal testen

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Echte AI toevoegen

Controleer eerst in de originele repository welke inferentiescripts, getrainde modelgewichten, preprocessing, en licentievoorwaarden beschikbaar zijn. Implementeer daarna een gevalideerde `predict(image)`-functie en voeg modelafhankelijke packages toe aan `requirements.txt`. Markeer detectieresultaten duidelijk als modelvoorspellingen die een menselijke inspectie vereisen.

Bronproject: https://github.com/AI-Toolbox-Waterkeringen/AI-toolbox-waterkeringen
