

import json
import threading
import time
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:

    def initialize(self, args):

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args["model_config"])

        using_decoupled = pb_utils.using_decoupled_model_transaction_policy(
            model_config
        )

        # Get IN configuration
        in_config = pb_utils.get_input_config_by_name(model_config, "IN")

        # Get OUT configuration
        out_config = pb_utils.get_output_config_by_name(model_config, "OUT")
        self.score_output_dtype = pb_utils.triton_string_to_numpy(
            out_config['data_type'])

        self.inflight_thread_count = 0
        self.inflight_thread_count_lck = threading.Lock()

    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. The request.get_response_sender() must be used to
        get an InferenceResponseSender object associated with the request.
        Use the InferenceResponseSender.send(response=<infer response object>,
        flags=<flags>) to send responses.

        In the final response sent using the response sender object, you must
        set the flags argument to TRITONSERVER_RESPONSE_COMPLETE_FINAL to
        indicate no responses will be sent for the corresponding request. If
        there is an error, you can set the error argument when creating a
        pb_utils.InferenceResponse. Setting the flags argument is optional and
        defaults to zero. When the flags argument is set to
        TRITONSERVER_RESPONSE_COMPLETE_FINAL providing the response argument is
        optional.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        None
        """

        # Visit individual request to start processing them. Note that execute
        # function is not required to wait for all the requests of the current
        # batch to be processed before returning.
        for request in requests:
            self.process_request(request)

        # Unlike in non-decoupled model transaction policy, execute function
        # here returns no response. A return from this function only notifies
        # Triton that the model instance is ready to receive another batch of
        # requests. As we are not waiting for the response thread to complete
        # here, it is possible that at any give time the model may be processing
        # multiple batches of requests. Depending upon the request workload,
        # this may lead to a lot of requests being processed by a single model
        # instance at a time. In real-world models, the developer should be
        # mindful of when to return from execute and be willing to accept next
        # request batch.
        return None

    def process_request(self, request):
        # Start a separate thread to send the responses for the request. The
        # sending back the responses is delegated to this thread.
        thread = threading.Thread(
            target=self.response_thread,
            args=(
                request.get_response_sender(),
                pb_utils.get_input_tensor_by_name(request, "IN").as_numpy(),
            ),
        )

        # A model using decoupled transaction policy is not required to send all
        # responses for the current request before returning from the execute.
        # To demonstrate the flexibility of the decoupled API, we are running
        # response thread entirely independent of the execute thread.
        thread.daemon = True

        with self.inflight_thread_count_lck:
            self.inflight_thread_count += 1

        thread.start()

    def response_thread(self, response_sender, in_input):
        # The response_sender is used to send response(s) associated with the
        # corresponding request.

        for idx in range(5):
            out_output = pb_utils.Tensor(
                "OUT", np.array([in_input[0]], dtype=self.score_output_dtype))
            response = pb_utils.InferenceResponse(
                output_tensors=[out_output])
            response_sender.send(response)

        # We must close the response sender to indicate to Triton that we are
        # done sending responses for the corresponding request. We can't use the
        # response sender after closing it. The response sender is closed by
        # setting the TRITONSERVER_RESPONSE_COMPLETE_FINAL.
        response_sender.send(
            flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL)

        with self.inflight_thread_count_lck:
            self.inflight_thread_count -= 1

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        Here we will wait for all response threads to complete sending
        responses.
        """

        print("Finalize invoked")

        inflight_threads = True
        cycles = 0
        logging_time_sec = 5
        sleep_time_sec = 0.1
        cycle_to_log = logging_time_sec / sleep_time_sec
        while inflight_threads:
            with self.inflight_thread_count_lck:
                inflight_threads = self.inflight_thread_count != 0
                if cycles % cycle_to_log == 0:
                    print(
                        f"Waiting for {self.inflight_thread_count} response threads to complete..."
                    )
            if inflight_threads:
                time.sleep(sleep_time_sec)
                cycles += 1

        print("Finalize complete...")
