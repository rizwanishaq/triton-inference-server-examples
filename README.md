---
title: "Triton Inference Server Series: Streaming with Python Backend"
description: Learn how to use the Python backend of NVIDIA Triton Inference Server to perform streaming inference.
image: "../../public/blogs/1_nautKVzRYdwRmIUtWJmlLw.png"
publishedAt: "2023-07-09"
updatedAt: "2023-07-09"
author: "Rizwan Ishaq"
isPublished: true
tags:
  - Python
  - Triton Inference Servering
  - Streaming
  - Machine learning
---

# Introduction

Are you tired of the traditional batch inference process? Are you ready to elevate your Python skills with real-time streaming inference? Look no further than the NVIDIA Triton Inference Server!

With Triton, you can leverage your Python expertise to deploy machine learning models in real time, directly from your GPU or CPU. Its streaming inference capabilities empower you to process data as it's generated, eliminating the need to wait for batches to accumulate.

In this comprehensive guide, I'll walk you through using the Python backend of Triton for streaming inference on images. Each step will be thoroughly explained, ensuring that even if you're new to Triton, you'll be able to get started with confidence.

## What is Triton Inference Server

According to [triton-inference-server](https://github.com/triton-inference-server/server):

> " **Triton Inference Server** is an open-source inference serving software that streamlines AI inferencing. Triton enables teams to deploy AI models from multiple deep learning and machine learning frameworks, including TensorRT, TensorFlow, PyTorch, ONNX, OpenVINO, Python, RAPIDS FIL, and more. Triton Inference Server supports inference across cloud, data center, edge, and embedded devices on NVIDIA GPUs, x86 and ARM CPUs, or AWS Inferentia. Triton Inference Server delivers optimized performance for various query types, including real-time, batched, ensembles, and audio/video streaming. Triton Inference Server is part of NVIDIA AI Enterprise, a software platform that accelerates the data science pipeline and streamlines the development and deployment of production AI.
> "

# Project Setup

Before we dive into setting up the server and client, let's ensure you're familiar with building and running Docker containers, as we'll be using it for the Triton Inference Server.

Assuming you have Docker installed and running, we can proceed with configuring the Python backend for streaming in the Triton Inference Server.

In Triton Inference Server, a specific folder structure is required for model inference. Begin by creating a folder named `models` (or any name of your choice). Within this folder, create a sub-folder named `jpg_model`, representing the model name. Inside `jpg_model`, create another folder named `1` to denote version 1. Additionally, include a `config.pbtxt` file for configuration details.

- **config.pbtxt**:

```plaintext
name: "jpg_model"
backend: "python"
max_batch_size: 0
input [
  {
    name: "image"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }
]
output [
  {
    name: "output_image"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }

]
instance_group [{count: 1, kind: KIND_CPU}]

sequence_batching {
  max_sequence_idle_microseconds: 60000000
  control_input {
    name: "START"
    control {
      kind: CONTROL_SEQUENCE_START
      int32_false_true: [0, 1]
    }
  }
  control_input {
    name: "READY"
    control {
      kind: CONTROL_SEQUENCE_READY
      int32_false_true: [0, 1]
    }
  }
  control_input {
    name: "END"
    control {
      kind: CONTROL_SEQUENCE_END
      int32_false_true: [0, 1]
    }
  }
  control_input {
    name: "CORRID"
    control {
      kind: CONTROL_SEQUENCE_CORRID
      data_type: TYPE_UINT64
    }
  }
}

```

The config.pbtxt file you provided contains essential configuration details for the Triton Inference Server, specifically for the jpg_model using a Python backend. Let's break down each section:

```plaintext
name: "jpg_model"
```

- Specifies the name of the model, in this case, it's jpg_model.

```plaintext
backend: "python"
```

- Defines the backend for this model. In this configuration, it's set to use the Python backend.

```plaintext
max_batch_size: 0
```

- Indicates the maximum batch size for the model. In this case, it's set to 0, meaning the model doesn't have a specific batch size constraint.

```plaintext
input [
  {
    name: "image"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }
]

```

- Describes the input configuration for the model. It specifies:
  - name: The name of the input, which is "image".
  - data_type: The data type of the input, which is TYPE_STRING.
  - dims: The dimensions of the input, in this case, it's set to a single dimension with size 1.

```plaintext
output [
  {
    name: "output_image"
    data_type: TYPE_STRING
    dims: [ 1 ]
  }
]
```

- Describes the output configuration for the model. It specifies:
  - name: The name of the output, which is "output_image".
  - data_type: The data type of the output, which is TYPE_STRING.
  - dims: The dimensions of the output, in this case, it's set to a single dimension with size 1.

```plaintext
instance_group [{count: 1, kind: KIND_CPU}]
```

- Defines the instance group configuration. In this case, it specifies one instance running on a CPU.

```plaintext
sequence_batching {
  max_sequence_idle_microseconds: 60000000
  control_input {
    name: "START"
    control {
      kind: CONTROL_SEQUENCE_START
      int32_false_true: [0, 1]
    }
  }
  control_input {
    name: "READY"
    control {
      kind: CONTROL_SEQUENCE_READY
      int32_false_true: [0, 1]
    }
  }
  control_input {
    name: "END"
    control {
      kind: CONTROL_SEQUENCE_END
      int32_false_true: [0, 1]
    }
  }
  control_input {
    name: "CORRID"
    control {
      kind: CONTROL_SEQUENCE_CORRID
      data_type: TYPE_UINT64
    }
  }
}
```

- Specifies sequence batching parameters. It includes:
  - max_sequence_idle_microseconds: The maximum time (in microseconds) a sequence can be idle before it's considered complete.
  - Control inputs (START, READY, END, CORRID) with their respective control types and data types.

This `config.pbtxt` file is crucial for configuring how Triton handles the input and output of the `jpg_model` with the Python backend. It defines the model's characteristics, including input and output details, instance group settings, and sequence batching behavior.

Once we have this, then inside the folder 1, we need to have model.py file

```python

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
```

The provided Python code is the implementation of a Triton Python Model for image processing. Let's break down each method and its purpose:

- `initialize(self, args: dict) -> None:` This method is called when the model is being initialized. It takes a dictionary args as input, which contains the model information. Specifically, it expects a key model_config which should be a JSON string containing the model configuration. In this method, the logger is initialized and the model configuration is loaded from the provided JSON string. Additionally, it retrieves the output image configuration and its data type.

- `execute(self, requests: list) -> list:` This method is responsible for processing inference requests. It takes a list of pb_utils.InferenceRequest objects as input, representing the batch of requests. It iterates through each request and checks for start and end flags. If a start or end flag is detected, it logs the action and creates an empty inference response. Otherwise, it processes the image, generates a response, and appends it to the list of responses.

- `finalize(self) -> None:` This method is called before the model is unloaded. In this case, it simply prints a message indicating that clean up is being performed.

This Python code defines a class `TritonPythonModel` that serves as a backend for Triton Inference Server. It contains methods for initialization, processing inference requests, and finalizing before unloading. The code demonstrates how to handle different aspects of inference processing, including logging, image manipulation, and response generation.

in this the `utils.py` file has following structure

```python
import base64
import cv2
import numpy as np


def mat_to_jpg_base64(mat: np.ndarray, quality: int = 95) -> str:
    """Converts a NumPy array representing an image to base64-encoded JPEG image data.

    Args:
        mat (np.ndarray): Input image as a NumPy array.
        quality (int, optional): Compression quality for JPEG encoding (0-100). Higher values mean better quality.
            Defaults to 95.

    Returns:
        str: Base64-encoded image data.
    """
    # Set JPEG compression quality
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

    # Encode the image
    encoded = cv2.imencode(".jpg", mat, encode_param)[1]
    b64 = base64.b64encode(encoded)
    return b64.decode('utf-8')


def jpg_base64_to_mat(jpg_base64_data: str) -> np.ndarray:
    """Converts base64-encoded JPEG image data to a NumPy array.

    Args:
        jpg_base64_data (str): Base64-encoded JPEG image data.

    Returns:
        np.ndarray: Image as a NumPy array.
    """
    img_string = base64.b64decode(jpg_base64_data)
    image = np.frombuffer(img_string, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image
```

The provided Python code in utils.py contains utility functions for converting images between NumPy arrays and base64-encoded JPEG format. Let's break down each function and its purpose:

- `mat_to_jpg_base64(mat: np.ndarray, quality: int = 95) -> str:` This function takes a NumPy array mat representing an image and an optional quality parameter for JPEG compression quality (default is 95). It converts the input image to a base64-encoded JPEG format and returns the resulting string. Here's how it works:

  - It first sets the JPEG compression quality using cv2.IMWRITE_JPEG_QUALITY.
  - It then encodes the image using cv2.imencode(), which returns a tuple where the second element is the encoded image data.
  - The encoded data is then base64 encoded and returned as a string.

- `jpg_base64_to_mat(jpg_base64_data: str) -> np.ndarray:` This function takes a base64-encoded JPEG image data as input. It decodes the base64 data to retrieve the binary image data, then converts it into a NumPy array representing an image. Here's how it works:
  - It decodes the base64 data using base64.b64decode() to get the binary image data.
  - It creates a NumPy array image from the binary data with data type np.uint8.
  - It then uses cv2.imdecode() to decode the image data, resulting in a NumPy array representing the image.

Once all this code is ready, we need to run the triton-inference server docker. Once the docker with the models is ready for serving, we need to send request from the client side. For that we use grpc, with python and that is the client code

These utility functions are crucial for converting images to and from base64-encoded JPEG format. They facilitate the processing of images in the Triton Inference Server, enabling efficient communication between the server and the backend.

```python
import cv2
import numpy as np
import subprocess
import logging
import argparse
from uuid import uuid4
from time import perf_counter
from typing import List, Tuple
from tritonclient.utils import np_to_triton_dtype
import tritonclient.grpc as grpcclient

from utils import unique_sequence_id, mat_to_jpg_base64, jpg_base64_to_mat


# Constants for Model Names and URLs
DEFAULT_MODEL_NAME = "jpg_model"
DEFAULT_TRITON_URL = "0.0.0.0:8001"

# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - Line:%(lineno)d -  %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('WebPStream')


class JpegStream:
    """
    A class for streaming video frames through a Triton Inference Server with a WebP compression model.

    Args:
        model_name (str, optional): Name of the WebP compression model. Defaults to "jpg_model".
        triton_url (str, optional): URL of the Triton Inference Server. Defaults to "0.0.0.0:8001".

    Attributes:
        triton_client (grpcclient.InferenceServerClient): Triton Inference Server client.
        model_name (str): Name of the WebP compression model.
        sequence_id (np.uint64): Unique ID for the sequence.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, triton_url: str = DEFAULT_TRITON_URL) -> None:
        """
        Initializes a WebPStream object.

        Args:
            model_name (str, optional): Name of the WebP compression model. Defaults to "jpg_model".
            triton_url (str, optional): URL of the Triton Inference Server. Defaults to "0.0.0.0:8001".
        """
        self.triton_client = grpcclient.InferenceServerClient(
            url=triton_url, verbose=False
        )
        self.model_name = model_name
        self.sequence_id = unique_sequence_id()

    def _input_output(self, image: np.ndarray) -> Tuple[List[grpcclient.InferInput], List[grpcclient.InferRequestedOutput]]:
        """
        Sets up input and output configuration for Triton Inference.

        Args:
            image (np.ndarray): Input image.

        Returns:
            tuple: Input and output configurations.
        """
        inputs = [
            grpcclient.InferInput(
                "image",
                image.shape,
                np_to_triton_dtype(image.dtype)
            ),
        ]
        outputs = [grpcclient.InferRequestedOutput("output_image")]
        inputs[0].set_data_from_numpy(image)

        return inputs, outputs

    def start(self, sequence_start: bool = True, sequence_end: bool = False) -> None:
        """
        Starts the streaming process.

        Args:
            sequence_start (bool, optional): Indicates if this is the start of a sequence. Defaults to True.
            sequence_end (bool, optional): Indicates if this is the end of a sequence. Defaults to False.
        """
        empty_image = np.array([1], dtype="object")

        inputs, _ = self._input_output(empty_image)

        _ = self.triton_client.infer(
            model_name=self.model_name,
            inputs=inputs,
            request_id=f'{uuid4()}',
            outputs=[],  # Empty response (no output expected for this request)
            sequence_id=self.sequence_id,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
        )
        logger.info(f'[✅] Stream: {self.sequence_id} - Started')

    def close(self, sequence_start: bool = False, sequence_end: bool = True) -> None:
        """
        Closes the streaming process.

        Args:
            sequence_start (bool, optional): Indicates if this is the start of a sequence. Defaults to False.
            sequence_end (bool, optional): Indicates if this is the end of a sequence. Defaults to True.
        """
        empty_image = np.array([1], dtype="object")

        inputs, _ = self._input_output(empty_image)

        _ = self.triton_client.infer(
            model_name=self.model_name,
            inputs=inputs,
            request_id=f'{uuid4()}',
            outputs=[],  # Empty response (no output expected for this request)
            sequence_id=self.sequence_id,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
        )
        logger.info(f'[✅] Stream: {self.sequence_id} - Closed')

    def __call__(self, image: np.ndarray, sequence_start: bool = False, sequence_end: bool = False) -> np.ndarray:
        """
        Compresses a single frame and returns the result.

        Args:
            image (np.ndarray): Input image.
            sequence_start (bool, optional): Indicates if this is the start of a sequence. Defaults to False.
            sequence_end (bool, optional): Indicates if this is the end of a sequence. Defaults to False.

        Returns:
            np.ndarray: Compressed output image.
        """
        image = mat_to_jpg_base64(image)
        image = np.array([image], dtype="object")

        inputs, outputs = self._input_output(image)

        response = self.triton_client.infer(
            model_name=self.model_name,
            inputs=inputs,
            request_id=f'{uuid4()}',
            outputs=outputs,
            sequence_id=self.sequence_id,
            sequence_start=sequence_start,
            sequence_end=sequence_end,
        )

        output_image = response.as_numpy("output_image")[0]
        output_image = jpg_base64_to_mat(output_image)

        return output_image

    def __enter__(self):
        """
        Method to enter the context.

        Returns:
            WebPStream: Returns the WebPStream object.
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Method to exit the context.

        Args:
            exc_type (type): Type of exception (if any).
            exc_value (Exception): The exception instance raised (if any).
            traceback (traceback): Traceback object (if any).
        """
        self.close()


def run(args: argparse.Namespace) -> None:
    """
    Runs the WebPStream with Triton Inference Server.

    Args:
        args (argparse.Namespace): Command line arguments.
    """
    if args.triton_server:
        triton_command = 'docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --gpus "device=0,1" --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/path-to-models-folder:/models docker-name tritonserver --model-repository=/models --backend-config=tensorflow,version=2'
        subprocess.run(triton_command, shell=True)

    video_stream = cv2.VideoCapture(args.video_file)

    with JpegStream(model_name=args.model_name, triton_url=args.triton_url) as stream:
        try:
            while True:
                ret, frame = video_stream.read()
                if not ret:
                    break

                start = perf_counter()  # Record start time
                output_image = stream(frame)
                logging.info(
                    f"[✅] Done in {(perf_counter() - start) * 1000:.2f}ms")

                cv2.imshow(f'{stream.sequence_id}', output_image)

                if cv2.waitKey(5) & 0xFF == 27:
                    break
        except (Exception, KeyboardInterrupt) as e:
            logger.exception(
                f'[❌] Stream: {stream.sequence_id} - Stopped - Error: {e}')
        finally:
            video_stream.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='WebPStream with Triton Inference Server')
    parser.add_argument('--model_name', type=str, default=DEFAULT_MODEL_NAME,
                        help='Name of the WebP compression model')
    parser.add_argument('--video_file', type=str,
                        default='Felipe.mp4', help='Path to the video file')
    parser.add_argument('--triton_url', type=str, default=DEFAULT_TRITON_URL,
                        help='URL of the Triton Inference Server')
    parser.add_argument('--triton_server', action='store_true',
                        help='Start the Triton Inference Server Docker')

    args = parser.parse_args()

    run(args)
```

Let's break down the code and provide detailed explanations for each part:

```python
import cv2
import numpy as np
import subprocess
import logging
import argparse
from uuid import uuid4
from time import perf_counter
from typing import List, Tuple
from tritonclient.utils import np_to_triton_dtype
import tritonclient.grpc as grpcclient

from utils import unique_sequence_id, mat_to_jpg_base64, jpg_base64_to_mat
```

- This section imports necessary libraries and modules:

  - cv2: OpenCV library for computer vision tasks.
  - numpy (np): Library for numerical operations in Python.
  - subprocess: Module for running shell commands from Python.
  - logging: Python's built-in logging module for generating log messages.
  - argparse: Library for parsing command-line arguments.
  - uuid4: Generates unique IDs (Universally Unique Identifiers).
  - perf_counter: Provides a high-resolution timer for measuring time intervals.
  - List, Tuple: Python typing for specifying data types.
  - np_to_triton_dtype, grpcclient: Modules for interacting with Triton Inference Server.
  - Custom utility functions from utils.py.

```python
# Constants for Model Names and URLs
DEFAULT_MODEL_NAME = "jpg_model"
DEFAULT_TRITON_URL = "0.0.0.0:8001"
```

- This section defines default values for the WebP compression model name (`DEFAULT_MODEL_NAME`) and Triton Inference Server URL (`DEFAULT_TRITON_URL`).

```python
# Configure the logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - Line:%(lineno)d -  %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('WebPStream')
```

- Here, the logger is configured to display log messages. It's set to show messages with a specific format, including a timestamp, log level, line number, and the actual message. The logger instance is named `WebPStream``.

```python
class JpegStream:
    # ... (class details)
```

- This defines the `JpegStream` class, which encapsulates the logic for streaming video frames through Triton Inference Server with WebP compression. This class contains methods for initializing, starting, closing, and processing frames.

```python
def run(args: argparse.Namespace) -> None:
    # ... (function details)
```

` The run function is the main entry point of the script. It takes command-line arguments (parsed by argparse) and orchestrates the operations. This includes starting the Triton server (if specified), opening a video stream, and processing frames through Triton Inference Server.

The provided code is designed to be run from the command line. It expects arguments like --model_name, --video_file, --triton_url, and --triton_server to be provided when running the script.

Overall, this script establishes a connection with the Triton server, processes video frames, sends them to the server for compression, and retrieves and displays the compressed frames. It allows for real-time video streaming with WebP compression using Triton Inference Server. Different model names, video sources, and Triton server URLs can be configured to adapt to various use cases.
