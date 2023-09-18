import base64
import cv2
import numpy as np
from PIL import Image
import io
import time


def unique_sequence_id() -> np.uint64:
    """
    Generates a unique sequence ID based on the current time.

    Returns:
        np.uint64: A unique sequence ID.
    """
    return np.uint64(time.time() * 1000000000)


def mat_to_webp_base64(mat: np.ndarray) -> str:
    """
    Converts a NumPy array (image in BGR format) to base64-encoded WebP image data.

    Args:
        mat (np.ndarray): NumPy array representing the image in BGR format.

    Returns:
        str: Base64-encoded WebP image data.
    """
    # Convert BGR to RGB
    mat_rgb = cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)

    pil_image = Image.fromarray(mat_rgb)
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


def mat_to_jpg_base64(mat: np.ndarray, quality: int = 95) -> str:
    """Converts a NumPy array representing an image to base64-encoded JPEG image data.

    Args:
        mat (np.ndarray): Input image as a NumPy array.
        quality (int, optional): Compression quality for JPEG encoding (0-100). Higher values mean better quality.
            Defaults to 95.

    Returns:
        str: Base64-encoded image data.
    """
    # Set JPEG compression quality
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

    # Encode the image
    encoded = cv2.imencode(".jpg", mat, encode_param)[1]
    b64 = base64.b64encode(encoded)
    return b64.decode('utf-8')


def jpg_base64_to_mat(jpg_base64_data: str) -> np.ndarray:
    """Converts base64-encoded JPEG image data to a NumPy array.

    Args:
        jpg_base64_data (str): Base64-encoded JPEG image data.

    Returns:
        np.ndarray: Image as a NumPy array.
    """
    img_string = base64.b64decode(jpg_base64_data)
    image = np.frombuffer(img_string, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image
