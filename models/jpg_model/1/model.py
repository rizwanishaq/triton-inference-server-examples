import numpy as np
import json
import triton_python_backend_utils as pb_utils
from utils import jpg_base64_to_mat, mat_to_jpg_base64
from typing import List


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
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

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])

        # Get outputs configuration for output image
        output_image_config = pb_utils.get_output_config_by_name(
            model_config, "output_image")

        # Convert Triton types to numpy types
        self.output_image_dtype = pb_utils.triton_string_to_numpy(
            output_image_config['data_type'])

    def execute(self, requests: List[pb_utils.InferenceRequest]) -> List[pb_utils.InferenceResponse]:
        """Process inference requests.

        Args:
            requests (List[pb_utils.InferenceRequest]): List of pb_utils.InferenceRequest.

        Returns:
            List[pb_utils.InferenceResponse]: List of pb_utils.InferenceResponse.
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
                image = image.as_numpy()[0].decode()

                image = jpg_base64_to_mat(image)

                out_image = mat_to_jpg_base64(image)

                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor(
                            "output_image",
                            np.array(
                                [out_image], dtype=self.output_image_dtype),
                        )
                    ])

            responses.append(inference_response)

        return responses

    def finalize(self) -> None:
        """Perform cleanups before model unloading."""
        print('Cleaning up...')
