# WebPStream with Triton Inference Server

This is a Python application for streaming video frames through a Triton Inference Server with a WebP compression model.

## Prerequisites

Before running this application, make sure you have the following installed:

- Python (3.6+)
- OpenCV
- NumPy
- Triton Inference Server
- Docker (optional, for starting Triton Inference Server using Docker)

## Getting Started

#### 1. Clone the Repository

```sh
git clone https://github.com/rizwanishaq/triton-inference-server-examples.git
```

#### 2. Install Dependencies

```sh
pip install opencv-python numpy tritonclient
```

#### 3. Start Triton Inference Server (if not already running)

If you don't have Triton Inference Server running, you can start it using Docker:

```sh
docker run --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --gpus "device=0,1" --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/absolute/path/to/models:/models dagan tritonserver --model-repository=/models --backend-config=tensorflow,version=2
```

#### 4. Run the Application

```sh
python main.py --model_name webp_model --video_file Felipe.mp4 --triton_url 0.0.0.0:8001 --triton_server
```

## Usage

```python
python main.py [--model_name MODEL_NAME] [--video_file VIDEO_FILE] [--triton_url TRITON_URL] [--triton_server]
```

- --model_name: Name of the WebP compression model (default: webp_model).
- --video_file: Path to the video file (default: Felipe.mp4).
- --triton_url: URL of the Triton Inference Server (default: 0.0.0.0:8001).
- --triton_server: Start the Triton Inference Server Docker (flag, default: False).

## Contributing

Pull requests and bug reports are welcome. For major changes, please open an issue first to discuss what you would like to change.
