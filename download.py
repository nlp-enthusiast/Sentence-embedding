from huggingface_hub import snapshot_download

snapshot_download(repo_id="bert-base-uncased", allow_patterns=["*.json",
                                                                         "pytorch_model.bin",
                                                                         "vocab.txt"], local_dir="./plm")