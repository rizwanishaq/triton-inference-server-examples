import argparse
import subprocess
import multiprocessing
from time import time

DEFAULT_MODEL_NAME = "webp_model"
DEFAULT_TRITON_URL = "0.0.0.0:8001"


def run_webp_stream(args):
    start_time = time()  # Record start time

    subprocess.run(["python", "webp_stream.py", "--model_name", args.model_name,
                   "--video_file", args.video_file, "--triton_url", args.triton_url])

    end_time = time()  # Record end time
    elapsed_time = end_time - start_time
    print(f"Process took {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='WebPStream Stress Test')
    parser.add_argument('--model_name', type=str, default=DEFAULT_MODEL_NAME,
                        help='Name of the WebP compression model')
    parser.add_argument('--video_file', type=str,
                        default='Felipe.mp4', help='Path to the video file')
    parser.add_argument('--triton_url', type=str, default=DEFAULT_TRITON_URL,
                        help='URL of the Triton Inference Server')
    parser.add_argument('--triton_server', action='store_true',
                        help='Start the Triton Inference Server Docker')
    parser.add_argument('--num_instances', type=int, default=4,
                        help='Number of instances to run concurrently')

    args = parser.parse_args()

    processes = []
    try:
        for _ in range(args.num_instances):
            process = multiprocessing.Process(
                target=run_webp_stream, args=(args,))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()
