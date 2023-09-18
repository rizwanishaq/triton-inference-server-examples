import base64
import cv2
import numpy as np


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
