import os
import json
import urllib.request
from fnmatch import fnmatch
from pathlib import Path
from collections import deque

import nibabel as nib
import wget

# ---------------------------------------------------------------------
# paths & constants (unchanged)
# ---------------------------------------------------------------------
resourcesdir = str(Path(__file__).parents[1]) + '/resources/'
downloads = resourcesdir + '/downloads/'
checkpointsdir = str(Path(__file__).parents[1]) + '/tutorials/checkpoints/'
repoID = '92p34'

# ---------------------------------------------------------------------
# lazy-loaded OSF index state (CHANGED: defer network work until needed)
# ---------------------------------------------------------------------
name_hashes = None          # set to dict after first load
glob_dict = None            # GlobDict(name_hashes) after first load
_INDEX_CACHE_PATH = Path(resourcesdir) / 'osf_index.json'

# ---------------------------------------------------------------------
# original tree walker (unchanged)
# ---------------------------------------------------------------------
def tree_osf_hashes(repoID, name_hashes, maxdepth=4, current_hash=''):
    """
    Iteratively fetch file hashes from the OSF repository using a queue to manage folders.
    """
    queue = deque([(0, '')])  # (depth, folder_hash)

    while queue:
        depth, folder_hash = queue.popleft()
        if depth > maxdepth:
            continue

        next_urls = deque([f"https://api.osf.io/v2/nodes/{repoID}/files/osfstorage/{folder_hash}"])
        try:
            while next_urls:
                next_url = next_urls.popleft()
                with urllib.request.urlopen(next_url) as url:
                    response = json.load(url)
                    data = response['data']

                    for item in data:
                        materialized_path = item['attributes']['materialized_path']
                        item_hash = item['attributes']['path']
                        kind = item['attributes']['kind']
                        name_hashes[materialized_path] = item_hash
                        if kind == 'folder' and depth + 1 <= maxdepth:
                            queue.append((depth + 1, item_hash))

                    nxt = response.get('links', {}).get('next', {}) if isinstance(response.get('links'), dict) else {}
                    if nxt:
                        next_urls.append(nxt)

        except Exception as e:
            print(f"Failed to fetch or parse data for URL {next_url}: {str(e)}")
            continue

# ---------------------------------------------------------------------
# utility for globbing (unchanged)
# ---------------------------------------------------------------------
class GlobDict(dict):
    def glob(self, match: str):
        return {k: v for k, v in self.items() if fnmatch(k, match)}

# ---------------------------------------------------------------------
# NEW: ensure index is loaded (from disk cache if possible), else fetch once
# ---------------------------------------------------------------------
def _ensure_index_loaded(force: bool = False):
    """
    Loads the OSF name->hash index and prepares glob_dict.
    - First tries on-disk cache at resources/osf_index.json (no internet).
    - If absent or 'force=True', queries OSF once and writes the cache.
    """
    global name_hashes, glob_dict

    if not force and name_hashes is not None and glob_dict is not None:
        return

    # try cache first (offline-friendly)
    if not force and _INDEX_CACHE_PATH.is_file():
        try:
            with open(_INDEX_CACHE_PATH, 'r') as f:
                loaded = json.load(f)
            if isinstance(loaded, dict) and loaded:
                name_hashes = loaded
                glob_dict = GlobDict(**name_hashes)
                return
        except Exception:
            pass  # fall through to online fetch

    # need to fetch online (only now)
    fetched = {}
    tree_osf_hashes(repoID, fetched, maxdepth=4, current_hash='')
    name_hashes = fetched
    glob_dict = GlobDict(**name_hashes)

    # write cache (best-effort)
    try:
        _INDEX_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_INDEX_CACHE_PATH, 'w') as f:
            json.dump(name_hashes, f)
    except Exception:
        pass

# ---------------------------------------------------------------------
# main API (same signatures)
# ---------------------------------------------------------------------
def get_map(repo='HippoMaps-initializationMaps', Dataset='*', Method='*', Modality='*',
            hemi='*', den='*', extension='.shape.gii'):
    """
    Search and load data from OSF with specified parameters.
    """
    # CHANGED: lazy-load OSF index here instead of on import
    _ensure_index_loaded()

    searchstr = f'/{repo}/Dataset-{Dataset}/{Method}-{Modality}*hemi-{hemi}*den-{den}*{extension}'
    globbed_results = glob_dict.glob(searchstr)

    data = []
    names = list(globbed_results.keys())

    for fn, h in globbed_results.items():
        outdir = Path(downloads + os.path.dirname(fn))
        outdir.mkdir(parents=True, exist_ok=True)
        outpath = Path(downloads + fn)
        if not outpath.is_file():
            wget.download(f"https://files.ca-1.osf.io/v1/resources/{repoID}/providers/osfstorage{h}",
                          out=str(outpath))
        data.append(nib.load(str(outpath)).darrays[0].data)

    return data, names


def get_tutorialCheckpoints(filenames):
    """
    Downloads checkpoint data needed for running hippomaps tutorials.
    filenames : list of strings
        Filenames to be downloaded
    """
    # CHANGED: lazy-load OSF index here instead of on import
    _ensure_index_loaded()

    Path(checkpointsdir).mkdir(parents=True, exist_ok=True)
    for fn in filenames:
        outpath = Path(checkpointsdir) / fn
        if not outpath.is_file():
            key = '/tutorialCheckpoints/' + fn
            if key not in name_hashes:
                raise FileNotFoundError(
                    f"'{fn}' not found in OSF index. Try _ensure_index_loaded(force=True) or check filename."
                )
            h = name_hashes[key]
            wget.download(f"https://files.ca-1.osf.io/v1/resources/{repoID}/providers/osfstorage{h}",
                          out=str(outpath))
