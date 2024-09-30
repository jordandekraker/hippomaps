import nibabel as nib
import urllib.request, json 
import wget
from fnmatch import fnmatch
from pathlib import Path
import os
resourcesdir = str(Path(__file__).parents[1]) + '/resources/'
downloads = resourcesdir + '/downloads/'
checkpointsdir = str(Path(__file__).parents[1]) + '/tutorials/checkpoints/'

repoID = '92p34'

# first get all available files (and their hash) from OSF

name_hashes={}
maxdepth=4
current_hash=''
def tree_osf_hashes(repoID, name_hashes, maxdepth, current_hash):
    """
    Recursively fetch file hashes from the OSF repository.

    Parameters
    ----------
    repoID : str
        The ID of the repository.
    name_hashes : dict
        A dictionary to store file names and their corresponding hashes.
    maxdepth : int
        The maximum depth to traverse the file tree.
    current_hash : str
        The hash of the current file or directory.

    Returns
    -------
    None
        Populates the name_hashes dictionary with file names and their corresponding hashes.
    """
    with urllib.request.urlopen(f"https://api.osf.io/v2/nodes/{repoID}/files/osfstorage{current_hash}") as url:
        data = json.load(url)['data']
        if isinstance(data, list):
            for item in data:
                materialized_path = item['attributes']['materialized_path']
                current_item_hash = item['attributes']['path']
                name_hashes[materialized_path] = current_item_hash
                # Recursively call for the next depth level
                if maxdepth > 1:
                    tree_osf_hashes(repoID, name_hashes, maxdepth-1, current_item_hash)
tree_osf_hashes(repoID,name_hashes,maxdepth,current_hash)

# utility for making dict searchable
class GlobDict(dict):
    def glob(self, match):
        """
       Glob-style pattern matching for the dictionary keys.@match should be a glob style pattern match (e.g. '*.txt')
       Parameters
       ----------
       match : str
           The glob-style pattern to match keys.
       Returns
       -------
       dict
           A filtered dictionary with keys matching the specified pattern.
        """
        return dict([(k,v) for k,v  in self.items() if fnmatch(k, match)])
glob_dict = GlobDict( **name_hashes )


# now the main function of interest
def get_map(repo='HippoMaps-initializationMaps',Dataset='*',Method='*',Modality='*', hemi='*', den='*', extension='.shape.gii'):
    """
    Search and load data from OSF with specified parameters.
    See https://osf.io/92p34/ for examples

    Parameters
    ----------
    repo : str, optional
        Top-level repository name (default: 'HippoMaps-initializationMaps').
    Dataset : str, optional
        Source of the map (default: '*').
    Method : str, optional
        Method being used (e.g., 'MRI', 'MRI-7T', or 'histology') (default: '*').
    Modality : str, optional
        Modality within the Method (e.g., 'qT1', 'FA', or 'Merker') (default: '*').
    hemi : str, optional
        Hemisphere (e.g., 'L', '*', or 'mix') (default: '*').
    den : str, optional
        Only search for data that is already at a given density (e.g., '0p5mm' or 'unfoldiso'). Note that data can always be resampled to a different density after loading (default: '*').
    extension : str, optional
        File extension (e.g., '.shape.gii', '.func.gii', or '.label.gii') (default: '.shape.gii').

    Returns
    -------
    data : list of arrays
        List of arrays containing data that matched search terms.
    names : list of str
        List of filenames that matched search terms.
    """
    searchstr = f'/{repo}/Dataset-{Dataset}/{Method}-{Modality}*hemi-{hemi}*den-{den}*{extension}'
    globbed_results = glob_dict.glob(searchstr)
    data = []
    names = [*globbed_results]
    for fn, h in globbed_results.items():
        if not os.path.isdir(downloads + os.path.dirname(fn)):
            os.makedirs(downloads + os.path.dirname(fn))
        if not os.path.isfile(downloads + fn):
            wget.download(f"https://files.ca-1.osf.io/v1/resources/{repoID}/providers/osfstorage{h}",out=downloads + fn)
        data.append(nib.load(downloads + fn).darrays[0].data)
    return data, names


def get_tutorialCheckpoints(filenames):
    """
    Downloads checkpoint data needed for running hippomaps tutorials. 
    filenames : list of strings
        Filenames to be downloaded
    """
    if not os.path.isdir(checkpointsdir):
        os.mkdir(checkpointsdir)
    for fn in filenames:
        if not os.path.isfile(checkpointsdir + fn):
            h = name_hashes['/tutorialCheckpoints/' + fn]
            wget.download(f"https://files.ca-1.osf.io/v1/resources/{repoID}/providers/osfstorage{h}",out=checkpointsdir + fn)