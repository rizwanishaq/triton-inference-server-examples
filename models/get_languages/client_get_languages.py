from tritonclient.utils import *
import numpy as np
import tritonclient.grpc as grpcclient
from uuid import uuid4


class TritonClient:
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
            url=url, verbose=True)
        self.model_name = model_name

    def infer(self):
        """
        Perform inference on a 30-second audio clip.

        Args:
            audio_path (str): Path to the audio file.
            language (np.ndarray): Language information.

        Returns:
            str: Transcription result.
        """

        languages_list = np.array([[""]], dtype="object")

        inputs = [
            grpcclient.InferInput(
                "languages_list",
                languages_list.shape,
                np_to_triton_dtype(languages_list.dtype)
            )
        ]
        outputs = [
            grpcclient.InferRequestedOutput("languages"),
        ]

        inputs[0].set_data_from_numpy(languages_list)

        result = self.triton_client.infer(
            model_name=self.model_name,
            inputs=inputs,
            request_id=f'{uuid4()}',
            outputs=[],
            parameters={'new_language': 'pn-IN'}
        )

        print(result.as_numpy('languages')[0].decode())


# Usage
if __name__ == "__main__":
    # Load audio and language

    client = TritonClient(url="0.0.0.0:8001", model_name="get_languages")
    transcription = client.infer()
