import os
from typing import List, Dict, Any
import numpy as np
import json
import triton_python_backend_utils as pb_utils
import fasttext
from utils import convert_to_language_code

fasttext.FastText.eprint = lambda x: None


class TritonPythonModel:
    """Python model for language identification using fastText.

    This class handles initialization, execution, and finalization of the model.

    Attributes:
        model (fasttext.FastText): Loaded fastText model.
        logger (pb_utils.Logger): Logger for model.
        model_config (Dict[str, Any]): Parsed model configuration.
        language_output_dtype (np.dtype): Data type for language output.
        score_output_dtype (np.dtype): Data type for score output.

    Methods:
        initialize(self, args: Dict[str, Any]) -> None:
            Called once during model loading for initialization.

        execute(self, requests: List[pb_utils.InferenceRequest]) -> List[pb_utils.InferenceResponse]:
            Called for processing inference requests.

        finalize(self) -> None:
            Called once when the model is being unloaded.

    """

    def initialize(self, args: Dict[str, Any]) -> None:
        """Initialize the model.

        Args:
            args (Dict[str, Any]): Dictionary containing model information.
                - model_config (str): JSON string containing the model configuration.
                - model_instance_kind (str): Model instance kind.
                - model_instance_device_id (str): Model instance device ID.
                - model_repository (str): Model repository path.
                - model_version (str): Model version.
                - model_name (str): Model name.
        """
        model_path = os.path.join(
            "/models", "t2t_lang", "1", "models", "models--facebook--fasttext-language-identification/snapshots/3af127d4124fc58b75666f3594bb5143b9757e78/model.bin"
        )

        self.model = fasttext.load_model(model_path)
        self.logger = pb_utils.Logger
        self.model_config = model_config = json.loads(args['model_config'])

        language_output_config = pb_utils.get_output_config_by_name(
            model_config, "language")
        self.language_output_dtype = pb_utils.triton_string_to_numpy(
            language_output_config['data_type'])

        score_output_config = pb_utils.get_output_config_by_name(
            model_config, "score")
        self.score_output_dtype = pb_utils.triton_string_to_numpy(
            score_output_config['data_type'])

    def execute(self, requests: List) -> List:
        """Process inference requests.

        Args:
            requests (List[pb_utils.InferenceRequest]): List of inference requests.

        Returns:
            List[pb_utils.InferenceResponse]: List of inference responses.
        """
        responses = []
        batch_input_text = []

        for request in requests:
            input_txt = pb_utils.get_input_tensor_by_name(request, "input_txt")
            input_txt = input_txt.as_numpy()[0][0].decode()
            batch_input_text.append(input_txt)

        batch_labels = self.model.predict(batch_input_text, k=1)

        for label, score in zip(batch_labels[0], batch_labels[1]):

            label = label[0].split("__label__")[1]
            label = convert_to_language_code(label)

            output_language_tensor = pb_utils.Tensor(
                "language",
                np.array([label], dtype=self.language_output_dtype),
            )

            output_score_tensor = pb_utils.Tensor(
                "score",
                np.array([score[0]], dtype=self.score_output_dtype),
            )

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_language_tensor, output_score_tensor]
            )

            responses.append(inference_response)

        return responses

    def finalize(self) -> None:
        """Perform cleanups before model unloading."""
        self.logger.log('Cleaning up...', self.logger.INFO)
