import numpy as np
import json
import triton_python_backend_utils as pb_utils
import time


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

        # self.metric_family = pb_utils.MetricFamily(
        #     name="requests_process_latency_ns",
        #     description="Cumulative time spent processing requests",
        #     kind=pb_utils.MetricFamily.COUNTER,  # or pb_utils.MetricFamily.GAUGE
        # )

        # self.metric = self.metric_family.Metric(
        #     labels={"model": "python_backend_batch", "version": "1"}
        # )

        self.logger = pb_utils.Logger
        self.model_config = model_config = json.loads(args['model_config'])
        output_image_config = pb_utils.get_output_config_by_name(
            model_config, "uid")

        # Convert Triton types to numpy types
        self.output_tensor_dtype = pb_utils.triton_string_to_numpy(
            output_image_config['data_type'])

    def execute(self, requests: list) -> list:
        """Process inference requests.

        Args:
            requests (list): List of pb_utils.InferenceRequest.

        Returns:
            list: List of pb_utils.InferenceResponse.
        """
        responses = []
        uid_batch = ''

        self.logger.log(f'Batch Size : {len(requests)}', self.logger.INFO)
        # Record the start time of processing the requests
        # start_ns = time.time_ns()
        for request in requests:
            uid = pb_utils.get_input_tensor_by_name(request, "uid")
            uid = uid.as_numpy()[0][0].decode()

            uid_batch = f'{uid_batch}_{uid}'

        for request in requests:
            uid = pb_utils.get_input_tensor_by_name(request, "uid")
            uid = uid.as_numpy()[0][0].decode()

            uid_return = f'{uid}_{uid_batch}'

            output_tensor = pb_utils.Tensor(
                "uid",
                np.array([uid_return], dtype=self.output_tensor_dtype),
            )

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            )

            responses.append(inference_response)

        # Record the end time of processing the requests
        # end_ns = time.time_ns()

        # Update metric to track cumulative requests processing latency.
        # There are three operations you can do with the Metric object:
        #   - Metric.increment(value): Increment the value of the metric by
        #       the given value. The type of the value is double. The 'COUNTER'
        #       kind does not support negative value.
        #   - Metric.set(value): Set the value of the metric to the given
        #       value. This operation is only supported in 'GAUGE' kind. The
        #       type of the value is double.
        #   - Metric.value(): Get the current value of the metric.
        # self.metric.increment(end_ns - start_ns)

        # self.logger.log_info(
        #     "Cumulative requests processing latency: {}".format(
        #         self.metric.value())
        # )

        return responses

    def finalize(self) -> None:
        """Perform cleanups before model unloading."""
        print('Cleaning up...')
