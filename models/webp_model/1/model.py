import numpy as np
import json
import triton_python_backend_utils as pb_utils
from utils import mat_to_webp_base64, webp_base64_to_mat

from typing import List


class TritonPythonModel:
    """Python model for WebP compression.

    This class handles initialization, execution, and finalization of the model.

    Attributes:
        model_config (dict): Parsed model configuration.

    Methods:
        initialize(self, args: dict) -> None:
            Called once during model loading for initialization.

        execute(self, requests: List[pb_utils.InferenceRequest]) -> List[pb_utils.InferenceResponse]:
            Called for processing inference requests.

        finalize(self) -> None:
            Called once when the model is being unloaded.

    """

    def initialize(self, args: dict) -> None:
        """Initialize the model.

        Args:
            args (dict): Dictionary containing model information.

                - model_config (str): JSON string containing the model configuration.
                - model_instance_kind (str): Model instance kind.
                - model_instance_device_id (str): Model instance device ID.
                - model_repository (str): Model repository path.
                - model_version (str): Model version.
                - model_name (str): Model name.
        """

        self.model_config = model_config = json.loads(args['model_config'])
        output_image_config = pb_utils.get_output_config_by_name(
            model_config, "output_image")

        # Convert Triton types to numpy types
        self.output_tensor_dtype = pb_utils.triton_string_to_numpy(
            output_image_config['data_type'])

    def execute(self, requests: List[pb_utils.InferenceRequest]) -> List[pb_utils.InferenceResponse]:
        """Process inference requests.

        Args:
            requests (list): List of pb_utils.InferenceRequest.

        Returns:
            list: List of pb_utils.InferenceResponse.
        """

        responses = []
        for request in requests:

            start_flag = pb_utils.get_input_tensor_by_name(
                request, "START").as_numpy()[0]
            end_flag = pb_utils.get_input_tensor_by_name(
                request, "END").as_numpy()[0]

            if end_flag or start_flag:
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[])

            else:
                image = pb_utils.get_input_tensor_by_name(request, "image")
                webp_data = image.as_numpy()[0]

                image = webp_base64_to_mat(webp_data)

                out_image = mat_to_webp_base64(image)

                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor(
                            "output_image",
                            np.array(
                                [out_image], dtype=self.output_tensor_dtype),
                        )
                    ])

            responses.append(inference_response)

        return responses

    def finalize(self) -> None:
        """Perform cleanups before model unloading."""
        print('Cleaning up...')
