import os

from datasets import load_dataset

# Set the mirror endpoint, just in case
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# Define the full path to the specific file you want
bench_file_url = "hf://datasets/Phineas476/EmbSpatial-Bench/embspatial_bench.json"

# Load the dataset using the 'data_files' argument
dataset = load_dataset("json", data_files=bench_file_url, split="train")

print("Benchmark dataset loaded successfully!")
print(dataset)
