import urllib.request, json 
import wget
from fnmatch import fnmatch
from pprint import pprint


repoID = 'v8acf'
name_hashes={}
maxdepth=4
current_hash=''
def tree_osf_hashes(repoID, name_hashes, maxdepth, current_hash):
    """
    Recursively fetches file hashes from the OSF repository. Verify with jordan
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
        The function populates the name_hashes dictionary with file names and their corresponding hashes.
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
tree_osf_hashes('v8acf',name_hashes,maxdepth,current_hash)

test=list(name_hashes.values())[-1]
# fn = wget.download(f"https://files.ca-1.osf.io/v1/resources/v8acf/providers/osfstorage{test}", out='test')
# for folders, add /?zip= to download

# get a flexible Dict reader
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
globbed_results = glob_dict.glob('*/tpl-upenn/*')

