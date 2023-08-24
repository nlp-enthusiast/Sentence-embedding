from .data import SentenceEmbeddingDataset as Dataset
from .data import NoDuplicatesDataLoader
from .evaluator import EmbeddingSimilarityEvaluator,SentenceEvaluator

import requests
from torch import Tensor, device
from typing import List, Callable
from tqdm.autonotebook import tqdm
import sys
import importlib
import os
import torch
from typing import Dict, Optional, Union
from pathlib import Path

import huggingface_hub
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from huggingface_hub import HfApi, hf_hub_url, cached_download, HfFolder
import fnmatch
from packaging import version

def cos_sim(a: Tensor, b: Tensor):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def http_get(url, path):
    """
    Downloads a URL to a given path on disc
    """
    if os.path.dirname(path) != '':
        os.makedirs(os.path.dirname(path), exist_ok=True)

    req = requests.get(url, stream=True)
    if req.status_code != 200:
        print("Exception when trying to download {}. Response {}".format(url, req.status_code), file=sys.stderr)
        req.raise_for_status()
        return

    download_filepath = path+"_part"
    with open(download_filepath, "wb") as file_binary:
        content_length = req.headers.get('Content-Length')
        total = int(content_length) if content_length is not None else None
        progress = tqdm(unit="B", total=total, unit_scale=True)
        for chunk in req.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                progress.update(len(chunk))
                file_binary.write(chunk)

    os.rename(download_filepath, path)
    progress.close()

def import_from_string(dotted_path):
    """
    Import a dotted module path and return the attribute/class designated by the
    last name in the path. Raise ImportError if the import failed.
    """
    try:
        module_path, class_name = dotted_path.rsplit('.', 1)
    except ValueError:
        msg = "%s doesn't look like a module path" % dotted_path
        raise ImportError(msg)

    try:
        module = importlib.import_module(dotted_path)
    except:
        module = importlib.import_module(module_path)

    try:
        return getattr(module, class_name)
    except AttributeError:
        msg = 'Module "%s" does not define a "%s" attribute/class' % (module_path, class_name)
        raise ImportError(msg)

def batch_to_device(batch, target_device: device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch

def fullname(o):
  """
  Gives a full name (package_name.class_name) for a class / object in Python. Will
  be used to load the correct classes from JSON files
  """

  module = o.__class__.__module__
  if module is None or module == str.__class__.__module__:
    return o.__class__.__name__  # Avoid reporting __builtin__
  else:
    return module + '.' + o.__class__.__name__


def snapshot_download(
        repo_id: str,
        revision: Optional[str] = None,
        cache_dir: Union[str, Path, None] = None,
        library_name: Optional[str] = None,
        library_version: Optional[str] = None,
        user_agent: Union[Dict, str, None] = None,
        ignore_files: Optional[List[str]] = None,
        use_auth_token: Union[bool, str, None] = None
) -> str:
    """
    Method derived from huggingface_hub.
    Adds a new parameters 'ignore_files', which allows to ignore certain files / file-patterns
    """
    if cache_dir is None:
        cache_dir = HUGGINGFACE_HUB_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    _api = HfApi()

    token = None
    if isinstance(use_auth_token, str):
        token = use_auth_token
    elif use_auth_token:
        token = HfFolder.get_token()

    model_info = _api.model_info(repo_id=repo_id, revision=revision, token=token)

    storage_folder = os.path.join(
        cache_dir, repo_id.replace("/", "_")
    )

    all_files = model_info.siblings
    # Download modules.json as the last file
    for idx, repofile in enumerate(all_files):
        if repofile.rfilename == "modules.json":
            del all_files[idx]
            all_files.append(repofile)
            break

    for model_file in all_files:
        if ignore_files is not None:
            skip_download = False
            for pattern in ignore_files:
                if fnmatch.fnmatch(model_file.rfilename, pattern):
                    skip_download = True
                    break

            if skip_download:
                continue

        url = hf_hub_url(
            repo_id, filename=model_file.rfilename, revision=model_info.sha
        )
        relative_filepath = os.path.join(*model_file.rfilename.split("/"))

        # Create potential nested dir
        nested_dirname = os.path.dirname(
            os.path.join(storage_folder, relative_filepath)
        )
        os.makedirs(nested_dirname, exist_ok=True)

        cached_download_args = {'url': url,
                                'cache_dir': storage_folder,
                                'force_filename': relative_filepath,
                                'library_name': library_name,
                                'library_version': library_version,
                                'user_agent': user_agent,
                                'use_auth_token': use_auth_token}

        if version.parse(huggingface_hub.__version__) >= version.parse("0.8.1"):
            # huggingface_hub v0.8.1 introduces a new cache layout. We sill use a manual layout
            # And need to pass legacy_cache_layout=True to avoid that a warning will be printed
            cached_download_args['legacy_cache_layout'] = True

        path = cached_download(**cached_download_args)

        if os.path.exists(path + ".lock"):
            os.remove(path + ".lock")

    return storage_folder