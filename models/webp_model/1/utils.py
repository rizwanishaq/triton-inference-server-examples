import cv2
import numpy as np
from PIL import Image
import io

import io
from PIL import Image
import cv2
import numpy as np
import base64


def mat_to_webp_base64(mat: np.ndarray) -> str:
    """
    Converts a NumPy array (image in BGR format) to base64-encoded WebP image data.

    Args:
        mat (np.ndarray): NumPy array representing the image in BGR format.

    Returns:
        str: Base64-encoded WebP image data.
    """
    pil_image = Image.fromarray(mat)
    webp_stream = io.BytesIO()
    pil_image.save(webp_stream, format="WEBP")

    webp_data = webp_stream.getvalue()
    return base64.b64encode(webp_data).decode('utf-8')


def webp_base64_to_mat(webp_base64_data: str) -> np.ndarray:
    """
    Converts base64-encoded WebP image data to a NumPy array in BGR format.

    Args:
        webp_base64_data (str): Base64-encoded WebP image data.

    Returns:
        np.ndarray: NumPy array representing the image in BGR format.
    """
    webp_data = base64.b64decode(webp_base64_data)
    webp_stream = io.BytesIO(webp_data)
    pil_image = Image.open(webp_stream)
    webp_image = np.array(pil_image)
    return cv2.cvtColor(webp_image, cv2.COLOR_RGB2BGR)
