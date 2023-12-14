import queue
from functools import partial
from typing import List, Optional

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype


class UserData:
    """User data class for managing completed requests in the stream."""

    def __init__(self) -> None:
        self._completed_requests: queue.Queue = queue.Queue()


def callback(user_data: UserData, result: Optional[grpcclient.InferResult], error: Optional[Exception]) -> None:
    """
    Callback function to handle the response or error from the async inference.

    Args:
        user_data (UserData): User-specific data object.
        result (Optional[grpcclient.InferResult]): Result of the inference.
        error (Optional[Exception]): Error, if any.

    Returns:
        None
    """
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


def main() -> None:
    """Main function for asynchronous inference."""
    model_name: str = "square_int32"
    in_data: np.ndarray = np.array(
        ["something I need to send again and again"], dtype="object")
    inputs: List[grpcclient.InferInput] = [grpcclient.InferInput(
        "IN", in_data.shape, np_to_triton_dtype(in_data.dtype))]
    outputs: List[grpcclient.InferRequestedOutput] = [
        grpcclient.InferRequestedOutput("OUT")]

    user_data: UserData = UserData()

    with grpcclient.InferenceServerClient(url="localhost:8001", verbose=True) as triton_client:
        # Establish stream
        triton_client.start_stream(callback=partial(callback, user_data))

        inputs[0].set_data_from_numpy(in_data)

        triton_client.async_stream_infer(
            model_name=model_name,
            inputs=inputs,
            request_id=str('1'),
            outputs=outputs,
            enable_empty_final_response=True
        )

        # Uncomment and use the necessary methods based on your requirements
        while True:
            data_item = user_data._completed_requests.get()
            closed = data_item.get_response().parameters.get(
                'triton_final_response').bool_param
            print(data_item.get_response(
            ).parameters.get('triton_final_response').bool_param)
            if closed:
                break


if __name__ == "__main__":
    main()
