from huggingface_hub import snapshot_download

snapshot_download(repo_id="sentence-transformers/all-MiniLM-L6-v2", allow_patterns=["*.json",
                                                                         "pytorch_model.bin",
                                                                         "vocab.txt"], local_dir="./all-MiniLM-L6-v2")