import json
import logging
import sys
from pathlib import Path

import requests

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Define FREQTRADE_DIR locally
FREQTRADE_DIR = Path(__file__).parent.parent

logger = logging.getLogger(__name__)

# Timeout for requests
req_timeout = 30


def clean_ui_subdir(directory: Path):
    if directory.is_dir():
        logger.info("Removing UI directory content.")

        for p in reversed(list(directory.glob("**/*"))):  # iterate contents from leaves to root
            if p.name in (".gitkeep", "fallback_file.html"):
                continue
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                p.rmdir()


def read_ui_version(dest_folder: Path) -> str | None:
    file = dest_folder / ".uiversion"
    if not file.is_file():
        return None

    with file.open("r") as f:
        return f.read()


def download_and_install_ui(dest_folder: Path, dl_url: str, version: str):
    from io import BytesIO
    from zipfile import ZipFile

    logger.info(f"Downloading {dl_url}")
    resp = requests.get(dl_url, timeout=req_timeout).content
    dest_folder.mkdir(parents=True, exist_ok=True)
    with ZipFile(BytesIO(resp)) as zf:
        for fn in zf.filelist:
            with zf.open(fn) as x:
                destfile = dest_folder / fn.filename
                if fn.is_dir():
                    destfile.mkdir(exist_ok=True)
                else:
                    destfile.write_bytes(x.read())
    with (dest_folder / ".uiversion").open("w") as f:
        f.write(version)


def get_ui_download_url(version: str | None, prerelease: bool) -> tuple[str, str]:
    """
    Get the download URL for the UI.
    :param version: Version to download
    :param prerelease: Allow prerelease versions
    :return: Download URL and version string
    """
    # base_url = "https://api.github.com/repos/freqtrade/frequi/"
    # # Get base UI Repo path
    # try:
    #     resp = requests.get(f"{base_url}releases", timeout=REQ_TIMEOUT)
    #     resp.raise_for_status()
    #     releases = resp.json()
    # except (requests.exceptions.RequestException, ValueError) as e:
    #     logger.warning("Could not fetch releases from github: %s. Fallback to ui_versions.json", e)
    fallback_file = FREQTRADE_DIR / 'commands' / 'ui_versions.json'
    with fallback_file.open() as f:
        releases = json.load(f)

    if version:
        tmp = [x for x in releases if x["name"] == version]
    else:
        tmp = [x for x in releases if prerelease or not x.get("prerelease")]

    if tmp:
        # Ensure we have the latest version
        if version is None:
            tmp.sort(key=lambda x: x["created_at"], reverse=True)
        latest_version = tmp[0]["name"]
        assets = tmp[0].get("assets", [])
    else:
        raise ValueError("UI-Version not found.")

    dl_url = ""
    if assets and len(assets) > 0:
        dl_url = assets[0]["browser_download_url"]

    # URL not found - try assets url
    if not dl_url and tmp:
        assets_url = tmp[0].get("assets_url")
        if assets_url:
            try:
                resp = requests.get(assets_url, timeout=req_timeout)
                resp.raise_for_status()
                assets = resp.json()
                if assets and len(assets) > 0:
                    dl_url = assets[0]["browser_download_url"]
            except (requests.exceptions.RequestException, ValueError) as e:
                logger.warning("Could not fetch assets from github: %s. No download URL found.", e)

    return dl_url, latest_version
