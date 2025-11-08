from huggingface_hub import snapshot_download
from huggingface_hub import model_info
from huggingface_hub import login
from transformers import AutoModel
from huggingface_hub import list_repo_files, hf_hub_url
import os
import requests

#myToken = "<YOUR_HUGGINGFACE_TOKEN>"
huggingface_key = os.getenv("HUGGINGFACE_KEY")
login(token=huggingface_key)
local_model_path = "../models/Mistral-7B-Instruct-v0.3"
model_name = "mistralai/Mistral-7B-Instruct-v0.3"

info = model_info(model_name)

for f in info.siblings:
    if f.size is not None:
        print(f"{f.rfilename} -> {f.size / (1024*1024):.2f} MB")
    else:
        print(f"{f.rfilename} -> Size unknown")



# repo_id = model_name
# files = list_repo_files(repo_id, token=huggingface_key)  # Add token=... if private
# # Step 2: Get file sizes
# for file_name in files:
#     try:
#         url = hf_hub_url(repo_id, filename=file_name)
#         r = requests.head(url)
#         size_mb = int(r.headers.get("Content-Length", 0)) / (1024*1024)
#         print(f"{file_name} -> {size_mb:.2f} MB")
#     except Exception as e:
#         print(f"{file_name} -> Could not get size ({e})")

# This will load the model to tell the parameter size, it takes time 
#model = AutoModel.from_pretrained(model_name)
#total_params = sum(p.numel() for p in model.parameters())
#print(f"Total parameters: {total_params:,}")  # e.g., 66,336,000

# This will start downloading the model to the specified local directory
snapshot_download(repo_id=model_name, local_dir=local_model_path)