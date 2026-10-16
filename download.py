from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="wikimedia/wikipedia",
    filename="20231101.simple/train-00000-of-00001.parquet",
    repo_type="dataset",
    local_dir="data",
)
print("Saved to:", path)