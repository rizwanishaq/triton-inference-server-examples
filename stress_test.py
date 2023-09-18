import argparse
import subprocess
import asyncio
import time
import threading
import multiprocessing

DEFAULT_MODEL_NAME = "webp_model"
DEFAULT_TRITON_URL = "0.0.0.0:8001"


def run_webp_stream(args):
    start_time = time.time()  # Record start time

    subprocess.run(["python", "webp_stream.py", "--model_name", args.model_name,
                   "--video_file", args.video_file, "--triton_url", args.triton_url])

    end_time = time.time()  # Record end time
    elapsed_time = end_time - start_time
    print(f"Connection took {elapsed_time:.2f} seconds using {args.method}")


async def main_asyncio(args):
    print("Using asyncio method")
    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(None, run_webp_stream, args)
             for _ in range(args.num_instances)]
    await asyncio.gather(*tasks)


def main_threading(args):
    print("Using threading method")
    threads = [threading.Thread(target=run_webp_stream, args=(args,))
               for _ in range(args.num_instances)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


def main_multiprocessing(args):
    print("Using multiprocessing method")
    processes = [multiprocessing.Process(
        target=run_webp_stream, args=(args,)) for _ in range(args.num_instances)]

    for process in processes:
        process.start()

    for process in processes:
        process.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='WebPStream Stress Test')
    parser.add_argument('--model_name', type=str, default=DEFAULT_MODEL_NAME,
                        help='Name of the WebP compression model')
    parser.add_argument('--video_file', type=str,
                        default='Felipe.mp4', help='Path to the video file')
    parser.add_argument('--triton_url', type=str, default=DEFAULT_TRITON_URL,
                        help='URL of the Triton Inference Server')
    parser.add_argument('--num_instances', type=int, default=4,
                        help='Number of instances to run concurrently')
    parser.add_argument('--use_threading', action='store_true',
                        help='Use threading')
    parser.add_argument('--use_multiprocessing', action='store_true',
                        help='Use multiprocessing')

    args = parser.parse_args()

    try:
        if not args.use_threading and not args.use_multiprocessing:
            args.method = "asyncio"
            asyncio.run(main_asyncio(args))
        elif args.use_threading:
            args.method = "threading"
            main_threading(args)
        elif args.use_multiprocessing:
            args.method = "multiprocessing"
            main_multiprocessing(args)
    except KeyboardInterrupt:
        pass
