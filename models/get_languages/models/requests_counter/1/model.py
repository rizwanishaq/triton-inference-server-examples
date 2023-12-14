import json
from typing import List, Dict
import triton_python_backend_utils as pb_utils
import numpy as np


class TritonPythonModel:
    """Python model for Triton inference server.

    Handles initialization, execution, and finalization of the model.

    Attributes:
        model_config (dict): Parsed model configuration.
        number_of_requests_dtype (numpy.dtype): Data type for the number of requests.
        request_counter (int): Counter for inference requests.
    """

    def initialize(self, args: Dict[str, str]) -> None:
        """Initialize the model.

        Args:
            args (dict): Dictionary containing model information.
                - model_config (str): JSON string containing the model configuration.
        """
        self.model_config = json.loads(args['model_config'])
        self.number_of_requests_dtype = self._get_number_of_requests_dtype()
        self.request_counter = 0

    def _get_number_of_requests_dtype(self) -> np.dtype:
        """Retrieve data type for the number of requests from the model configuration."""
        config_name = "number_of_requests"
        number_of_requests_config = pb_utils.get_output_config_by_name(
            self.model_config, config_name
        )
        return pb_utils.triton_string_to_numpy(number_of_requests_config['data_type'])

    def _get_request_type(self, request) -> str:
        """Retrieve the request type from the input tensor named 'request_type'."""
        return pb_utils.get_input_tensor_by_name(request, "request_type").as_numpy()[0].decode()

    def _process_request(self, request_type: str):
        """Process the inference request based on the request type."""
        if request_type == "START":
            self.request_counter += 1
            return pb_utils.InferenceResponse(output_tensors=[])
        elif request_type == "END":
            self.request_counter -= 1
            return pb_utils.InferenceResponse(output_tensors=[])
        else:
            return pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor(
                    "number_of_requests",
                    np.array(self.request_counter,
                             self.number_of_requests_dtype)
                )
            ])

    def execute(self, requests: List) -> List:
        """Process inference requests.

        Args:
            requests (list): List of pb_utils.InferenceRequest.

        Returns:
            list: List of pb_utils.InferenceResponse.
        """
        responses = []

        for request in requests:
            request_type = self._get_request_type(request)
            response = self._process_request(request_type)
            responses.append(response)

        return responses

    def finalize(self) -> None:
        """Perform cleanups before model unloading."""
        print('Cleaning up...')
