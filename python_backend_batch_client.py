import cv2
import numpy as np
import subprocess
import logging
from uuid import uuid4
from tritonclient.utils import np_to_triton_dtype
import tritonclient.grpc as grpcclient
from typing import List
import threading
from concurrent.futures import ThreadPoolExecutor

# Constants for Model Names and URLs
DEFAULT_MODEL_NAME = "python_backend_batch"
DEFAULT_TRITON_URL = "0.0.0.0:8001"


def client_thread(client: "TritonClient", uid: str) -> None:
    """
    Executes Triton client for a specific UID.

    Args:
        client (TritonClient): The Triton client object.
        uid (str): The UID for processing.
    """
    result = client(uid)
    print(f"Output for {uid}: {result}")


class TritonClient:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, triton_url: str = DEFAULT_TRITON_URL) -> None:
        """
        Initializes the Triton client.

        Args:
            model_name (str, optional): Name of the model. Defaults to "python_backend_batch".
            triton_url (str, optional): URL of the Triton Inference Server. Defaults to "0.0.0.0:8001".
        """
        self.triton_client = grpcclient.InferenceServerClient(
            url=triton_url, verbose=False)
        self.model_name = model_name

    def __call__(self, uid: str) -> str:
        """
        Calls the Triton client to process a UID.

        Args:
            uid (str): The UID for processing.

        Returns:
            str: Processed UID.
        """
        uid = np.array([[uid]], dtype="object")

        inputs = [
            grpcclient.InferInput(
                "uid",
                uid.shape,
                np_to_triton_dtype(uid.dtype)
            ),
        ]
        outputs = [grpcclient.InferRequestedOutput("uid")]
        inputs[0].set_data_from_numpy(uid)

        response = self.triton_client.infer(
            model_name=self.model_name,
            inputs=inputs,
            request_id=f'{uuid4()}',
            outputs=outputs,
        )

        output_uid = response.as_numpy("uid")[0]
        return output_uid


if __name__ == "__main__":
    # Create 20 clients and corresponding UIDs
    clients = [TritonClient() for _ in range(20)]
    uids = [f'uid_{i}' for i in range(20)]

    # Use ThreadPoolExecutor for concurrent execution
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(client_thread, client, uid)
                   for client, uid in zip(clients, uids)]

        # Wait for all threads to finish
        for future in futures:
            future.result()
