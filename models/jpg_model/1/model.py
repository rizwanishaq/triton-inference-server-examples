import json
import triton_python_backend_utils as pb_utils
from utils import jpg_base64_to_mat, mat_to_jpg_base64
import numpy as np


class TritonPythonModel:
    """Triton Python Model for image processing."""

    def initialize(self, args: dict) -> None:
        """Initialize the model.

        Args:
            args (dict): Dictionary containing model information.

                - model_config (str): JSON string containing the model configuration.
        """

        # Initialize the logger and model configuration
        self.logger = pb_utils.Logger
        self.model_config = json.loads(args['model_config'])

        # Get output image configuration and data type
        output_image_config = pb_utils.get_output_config_by_name(
            self.model_config, "output_image")
        self.output_image_dtype = pb_utils.triton_string_to_numpy(
            output_image_config['data_type'])

    def execute(self, requests: list) -> list:
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
                # Get sequence ID and log the action
                sequence_id = pb_utils.get_input_tensor_by_name(
                    request, "CORRID").as_numpy()[0]
                action = "started" if start_flag else "stopped"
                self.logger.log(
                    f'Stream {sequence_id} {action}', self.logger.INFO)
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[])
            else:
                # Process the image and create response
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
