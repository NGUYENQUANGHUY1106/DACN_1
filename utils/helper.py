from PIL import Image
import io

def load_image(uploaded_file):
    image_data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    return Image.open(io.BytesIO(image_data))