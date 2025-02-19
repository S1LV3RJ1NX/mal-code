import os
from pathlib import Path

from huggingface_hub import HfApi

HF_TOKEN = os.getenv("HF_TOKEN")


def upload_models_to_hf(repo_id: str, weights_dir: str):
    """
    Upload model checkpoints to Hugging Face Hub.

    Args:
        repo_id (str): The Hugging Face repo ID in format 'username/repo-name'
        weights_dir (str): Local directory containing the model weights files
    """
    # Initialize Hugging Face API
    api = HfApi(token=HF_TOKEN)

    # Check if repo exists, if not create it
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
    except Exception:
        print(f"Repository {repo_id} does not exist. Creating it...")
        api.create_repo(repo_id=repo_id, repo_type="model", private=False)
        print(f"Created repository {repo_id}")

    # Get all .pt files from weights directory
    weights_path = Path(weights_dir)
    model_files = list(weights_path.glob("*.pt"))

    if not model_files:
        print(f"No .pt files found in {weights_dir}")
        return

    print(f"Found {len(model_files)} model files")

    # Upload each file
    for model_file in model_files:
        print(f"Uploading {model_file.name}...")

        try:
            api.upload_file(
                path_or_fileobj=str(model_file),
                path_in_repo=model_file.name,
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"Successfully uploaded {model_file.name}")

        except Exception as e:
            print(f"Error uploading {model_file.name}: {str(e)}")


if __name__ == "__main__":
    # Example usage
    repo_id = "s1lv3rj1nx/ch2"  # Replace with your repo
    weights_dir = "/home/jovyan/mal-code/ch2/checkpoints/"  # Replace with your weights directory

    upload_models_to_hf(repo_id, weights_dir)
