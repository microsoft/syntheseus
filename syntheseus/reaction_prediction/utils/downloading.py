import json
import logging
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__file__)


def get_cache_dir(key: str) -> Path:
    """Get the cache directory for a given key (e.g. model name)."""

    # First, check if the cache directory has been manually overriden.
    cache_dir_from_env = os.getenv("SYNTHESEUS_CACHE_DIR")
    if cache_dir_from_env is not None:
        # If yes, use the path provided.
        cache_dir = Path(cache_dir_from_env)
    else:
        # If not, construct a reasonable default.
        cache_dir = Path(os.getenv("HOME", ".")) / ".cache" / "torch" / "syntheseus"

    cache_dir = cache_dir / key
    cache_dir.mkdir(parents=True, exist_ok=True)

    return cache_dir


def get_figshare_download_link(figshare_id: int) -> str:
    """Query Figshare API to get a download link for a given file ID."""

    logger.info(f"Downloading metadata for Figshare ID {figshare_id}")
    with urllib.request.urlopen(
        f"https://api.figshare.com/v2/articles/{figshare_id}/files"
    ) as response:
        [metadata] = json.load(response)

    URL_KEY = "download_url"

    if URL_KEY not in metadata:
        raise ValueError(f"Got corrupted Figshare metadata: {metadata}")

    return metadata[URL_KEY]


def get_cache_dir_download_if_missing(key: str, figshare_id: int) -> Path:
    """Get the cache directory for a given key, but populate from Figshare if empty."""

    cache_dir = get_cache_dir(key)
    cache_dir_contents = list(cache_dir.iterdir())

    MODEL_ZIP_FILE_NAME = "model.zip"

    if len(cache_dir_contents) == 1 and cache_dir_contents[0].name == MODEL_ZIP_FILE_NAME:
        # It seems either a previous download was interrupted, or there was an issue during
        # extraction. Either way, remove the zip file to trigger a re-download.
        cache_dir_contents[0].unlink()
        cache_dir_contents = []

    if not cache_dir_contents:
        cache_zip_path = cache_dir / MODEL_ZIP_FILE_NAME
        link = get_figshare_download_link(figshare_id)

        logger.info(f"Downloading data from {link} to {cache_zip_path}")
        urllib.request.urlretrieve(link, cache_zip_path)

        with zipfile.ZipFile(cache_zip_path, "r") as f_zip:
            f_zip.extractall(cache_dir)

        cache_zip_path.unlink()

    return cache_dir


def get_default_model_dir_from_cache(model_name: str, is_forward: bool) -> Optional[Path]:
    default_checkpoint_ids_file_path = (
        Path(__file__).parent.parent / "inference" / "default_checkpoint_ids.yml"
    )

    if not default_checkpoint_ids_file_path.exists():
        logger.info(
            f"Could not obtain a default model link: {default_checkpoint_ids_file_path} does not exist"
        )
        return None

    with open(default_checkpoint_ids_file_path, "rt") as f_defaults:
        default_checkpoint_ids = yaml.safe_load(f_defaults)

    assert default_checkpoint_ids.keys() == {"backward", "forward"}

    forward_backward_key = "forward" if is_forward else "backward"
    checkpoint_ids = default_checkpoint_ids[forward_backward_key]

    if model_name not in checkpoint_ids:
        logger.info(f"Could not obtain a default model link: no entry for {model_name}")
        return None

    return get_cache_dir_download_if_missing(
        f"{model_name}_{forward_backward_key}", figshare_id=checkpoint_ids[model_name]
    )
