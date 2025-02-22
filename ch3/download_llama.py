# To download the model, you need to have a Hugging Face account and a token.
# You can get the token from https://huggingface.co/settings/tokens
# Make sure you have access to the model.
# Set the HF_TOKEN environment variable in .env file
import os

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

snapshot_download(
    repo_id="meta-llama/Llama-3.2-3B",
    revision="main",
    local_dir="./pretrained-weights",
    token=HF_TOKEN,
)
