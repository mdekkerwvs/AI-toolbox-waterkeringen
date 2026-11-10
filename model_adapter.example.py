"""Illustrative adapter contract. NOT an actual crack detection model.

After identifying and installing the upstream AI Toolbox detection code and
obtaining suitable trained model weights, create a separate file named
model_adapter.py. It must implement predict_mask(image: PIL.Image.Image)
returning a boolean/0-1 array of shape (image.height, image.width).
Do NOT rename this example to model_adapter.py: it intentionally raises.
"""

def predict_mask(image):
    raise NotImplementedError('Nog geen getraind scheurdetectiemodel geconfigureerd.')
