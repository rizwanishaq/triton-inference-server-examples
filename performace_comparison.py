import time
import subprocess

# Number of iterations to test
num_iterations = 1

# Measure the execution time for webp_stream.py
webp_start_time = time.time()
for _ in range(num_iterations):
    subprocess.run(["python3", "webp_stream.py", "--model_name",
                   "webp_model", "--video_file", "Felipe.mp4"])
webp_end_time = time.time()

# Measure the execution time for jpg_stream.py
jpg_start_time = time.time()
for _ in range(num_iterations):
    subprocess.run(["python3", "jpg_stream.py", "--model_name",
                   "jpg_model", "--video_file", "Felipe.mp4"])
jpg_end_time = time.time()

# Calculate the average time per iteration for each script
webp_average_time = (webp_end_time - webp_start_time) / num_iterations
jpg_average_time = (jpg_end_time - jpg_start_time) / num_iterations

# Print the results
print(
    f"Average execution time per iteration for webp_stream.py: {webp_average_time:.2f} seconds")
print(
    f"Average execution time per iteration for jpg_stream.py: {jpg_average_time:.2f} seconds")
