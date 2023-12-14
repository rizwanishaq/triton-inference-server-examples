from tritonclient.utils import *
import numpy as np
import tritonclient.grpc as grpcclient
from uuid import uuid4


class LoadService:
    """
    TritonClient class for interacting with Triton Inference Server.
    """

    def __init__(self, url, model_name):
        """
        Initialize TritonClient.

        Args:
            url (str): URL of Triton server.
            model_name (str): Name of the model.
        """
        self.triton_client = grpcclient.InferenceServerClient(
            url=url, verbose=False)
        self.model_name = model_name

    def increment(self):
        """
        Perform inference on a 30-second audio clip.

        Args:
            audio_path (str): Path to the audio file.
            language (np.ndarray): Language information.

        Returns:
            str: Transcription result.
        """

        request_type = np.array(["START"], dtype="object")

        inputs = [
            grpcclient.InferInput(
                "request_type",
                request_type.shape,
                np_to_triton_dtype(request_type.dtype)
            )
        ]

        inputs[0].set_data_from_numpy(request_type)

        _ = self.triton_client.infer(
            model_name=self.model_name,
            inputs=inputs,
            request_id=f'{uuid4()}',
            outputs=[],
        )

    def decriment(self):
        """
        Perform inference on a 30-second audio clip.

        Args:
            audio_path (str): Path to the audio file.
            language (np.ndarray): Language information.

        Returns:
            str: Transcription result.
        """

        request_type = np.array(["END"], dtype="object")

        inputs = [
            grpcclient.InferInput(
                "request_type",
                request_type.shape,
                np_to_triton_dtype(request_type.dtype)
            )
        ]

        inputs[0].set_data_from_numpy(request_type)

        _ = self.triton_client.infer(
            model_name=self.model_name,
            inputs=inputs,
            request_id=f'{uuid4()}',
            outputs=[],
        )

    def get_load(self):
        """
        Perform inference on a 30-second audio clip.

        Args:
            audio_path (str): Path to the audio file.
            language (np.ndarray): Language information.

        Returns:
            str: Transcription result.
        """

        request_type = np.array(["load"], dtype="object")

        inputs = [
            grpcclient.InferInput(
                "request_type",
                request_type.shape,
                np_to_triton_dtype(request_type.dtype)
            )
        ]
        outputs = [
            grpcclient.InferRequestedOutput("number_of_requests"),
        ]

        inputs[0].set_data_from_numpy(request_type)

        response = self.triton_client.infer(
            model_name=self.model_name,
            inputs=inputs,
            request_id=f'{uuid4()}',
            outputs=outputs,
        )

        return response.as_numpy('number_of_requests')


# Usage
if __name__ == "__main__":
    # Load audio and language

    client = LoadService(url="172.26.161.44:8001",
                         model_name="requests_counter")
    response = client.get_load()
    print(response)
