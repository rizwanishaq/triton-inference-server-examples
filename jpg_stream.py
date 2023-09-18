import cv2
import numpy as np
import subprocess
import logging
import argparse
from uuid import uuid4
from time import perf_counter
from typing import List, Tuple
from tritonclient.utils import np_to_triton_dtype
import tritonclient.grpc as grpcclient

from utils import unique_sequence_id, mat_to_jpg_base64, jpg_base64_to_mat


# Constants for Model Names and URLs
DEFAULT_MODEL_NAME = "jpg_model"
DEFAULT_TRITON_URL = "0.0.0.0:8001"

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - Line:%(lineno)d -  %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('WebPStream')


class JpegStream:
    """
    A class for streaming video frames through a Triton Inference Server with a WebP compression model.

    Args:
        model_name (str, optional): Name of the WebP compression model. Defaults to "jpg_model".
        triton_url (str, optional): URL of the Triton Inference Server. Defaults to "0.0.0.0:8001".

    Attributes:
        triton_client (grpcclient.InferenceServerClient): Triton Inference Server client.
        model_name (str): Name of the WebP compression model.
        sequence_id (np.uint64): Unique ID for the sequence.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, triton_url: str = DEFAULT_TRITON_URL) -> None:
        """
        Initializes a WebPStream object.

        Args:
            model_name (str, optional): Name of the WebP compression model. Defaults to "jpg_model".
            triton_url (str, optional): URL of the Triton Inference Server. Defaults to "0.0.0.0:8001".
        """
        self.triton_client = grpcclient.InferenceServerClient(
            url=triton_url, verbose=False
        )
        self.model_name = model_name
        self.sequence_id = unique_sequence_id()

    def _input_output(self, image: np.ndarray) -> Tuple[List[grpcclient.InferInput], List[grpcclient.InferRequestedOutput]]:
        """
        Sets up input and output configuration for Triton Inference.

        Args:
            image (np.ndarray): Input image.

        Returns:
            tuple: Input and output configurations.
        """
        inputs = [
            grpcclient.InferInput(
                "image",
                image.shape,
                np_to_triton_dtype(image.dtype)
            ),
        ]
        outputs = [grpcclient.InferRequestedOutput("output_image")]
        inputs[0].set_data_from_numpy(image)

        return inputs, outputs

    def start(self, sequence_start: bool = True, sequence_end: bool = False) -> None:
        """
        Starts the streaming process.

        Args:
            sequence_start (bool, optional): Indicates if this is the start of a sequence. Defaults to True.
            sequence_end (bool, optional): Indicates if this is the end of a sequence. Defaults to False.
        """
        empty_image = np.array([1], dtype="object")

        inputs, _ = self._input_output(empty_image)

        _ = self.triton_client.infer(
            model_name=self.model_name,
            inputs=inputs,
            request_id=f'{uuid4()}',
            outputs=[],  # Empty response (no output expected for this request)
            sequence_id=self.sequence_id,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
        )
        logger.info(f'[✅] Stream: {self.sequence_id} - Started')

    def close(self, sequence_start: bool = False, sequence_end: bool = True) -> None:
        """
        Closes the streaming process.

        Args:
            sequence_start (bool, optional): Indicates if this is the start of a sequence. Defaults to False.
            sequence_end (bool, optional): Indicates if this is the end of a sequence. Defaults to True.
        """
        empty_image = np.array([1], dtype="object")

        inputs, _ = self._input_output(empty_image)

        _ = self.triton_client.infer(
            model_name=self.model_name,
            inputs=inputs,
            request_id=f'{uuid4()}',
            outputs=[],  # Empty response (no output expected for this request)
            sequence_id=self.sequence_id,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
        )
        logger.info(f'[✅] Stream: {self.sequence_id} - Closed')

    def __call__(self, image: np.ndarray, sequence_start: bool = False, sequence_end: bool = False) -> np.ndarray:
        """
        Compresses a single frame and returns the result.

        Args:
            image (np.ndarray): Input image.
            sequence_start (bool, optional): Indicates if this is the start of a sequence. Defaults to False.
            sequence_end (bool, optional): Indicates if this is the end of a sequence. Defaults to False.

        Returns:
            np.ndarray: Compressed output image.
        """
        image = mat_to_jpg_base64(image)
        image = np.array([image], dtype="object")

        inputs, outputs = self._input_output(image)

        response = self.triton_client.infer(
            model_name=self.model_name,
            inputs=inputs,
            request_id=f'{uuid4()}',
            outputs=outputs,
            sequence_id=self.sequence_id,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
        )

        output_image = response.as_numpy("output_image")[0]
        output_image = jpg_base64_to_mat(output_image)

        return output_image

    def __enter__(self):
        """
        Method to enter the context.

        Returns:
            WebPStream: Returns the WebPStream object.
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Method to exit the context.

        Args:
            exc_type (type): Type of exception (if any).
            exc_value (Exception): The exception instance raised (if any).
            traceback (traceback): Traceback object (if any).
        """
        self.close()


def run(args: argparse.Namespace) -> None:
    """
    Runs the WebPStream with Triton Inference Server.

    Args:
        args (argparse.Namespace): Command line arguments.
    """
    if args.triton_server:
        triton_command = 'docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --gpus "device=0,1" --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/path-to-models-folder:/models docker-name tritonserver --model-repository=/models --backend-config=tensorflow,version=2'
        subprocess.run(triton_command, shell=True)

    video_stream = cv2.VideoCapture(args.video_file)

    with JpegStream(model_name=args.model_name, triton_url=args.triton_url) as stream:
        try:
            while True:
                ret, frame = video_stream.read()
                if not ret:
                    break

                start = perf_counter()  # Record start time
                output_image = stream(frame)
                logging.info(
                    f"[✅] Done in {(perf_counter() - start) * 1000:.2f}ms")

                cv2.imshow(f'{stream.sequence_id}', output_image)

                if cv2.waitKey(5) & 0xFF == 27:
                    break
        except (Exception, KeyboardInterrupt) as e:
            logger.exception(
                f'[❌] Stream: {stream.sequence_id} - Stopped - Error: {e}')
        finally:
            video_stream.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='WebPStream with Triton Inference Server')
    parser.add_argument('--model_name', type=str, default=DEFAULT_MODEL_NAME,
                        help='Name of the WebP compression model')
    parser.add_argument('--video_file', type=str,
                        default='Felipe.mp4', help='Path to the video file')
    parser.add_argument('--triton_url', type=str, default=DEFAULT_TRITON_URL,
                        help='URL of the Triton Inference Server')
    parser.add_argument('--triton_server', action='store_true',
                        help='Start the Triton Inference Server Docker')

    args = parser.parse_args()

    run(args)
