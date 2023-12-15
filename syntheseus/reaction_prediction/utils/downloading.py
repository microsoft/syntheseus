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


def get_cache_dir_download_if_missing(key: str, link: str) -> Path:
    """Get the cache directory for a given key, but populate by downloading from link if empty."""

    cache_dir = get_cache_dir(key)
    if not any(cache_dir.iterdir()):
        cache_zip_path = cache_dir / "model.zip"

        logger.info(f"Downloading data from {link} to {cache_zip_path}")
        urllib.request.urlretrieve(link, cache_zip_path)

        with zipfile.ZipFile(cache_zip_path, "r") as f_zip:
            f_zip.extractall(cache_dir)

        cache_zip_path.unlink()

    return cache_dir


def get_default_model_dir_from_cache(model_name: str, is_forward: bool) -> Optional[Path]:
    default_model_links_file_path = (
        Path(__file__).parent.parent / "inference" / "default_checkpoint_links.yml"
    )

    if not default_model_links_file_path.exists():
        logger.info(
            f"Could not obtain a default model link: {default_model_links_file_path} does not exist"
        )
        return None

    with open(default_model_links_file_path, "rt") as f_defaults:
        default_model_links = yaml.safe_load(f_defaults)

    assert default_model_links.keys() == {"backward", "forward"}

    forward_backward_key = "forward" if is_forward else "backward"
    model_links = default_model_links[forward_backward_key]

    if model_name not in model_links:
        logger.info(f"Could not obtain a default model link: no entry for {model_name}")
        return None

    return get_cache_dir_download_if_missing(
        f"{model_name}_{forward_backward_key}", link=model_links[model_name]
    )
