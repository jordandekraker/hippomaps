# Configuration file for the Sphinx documentation builder.
# path to the root of hippomaps  relative to the documentation root
import os
import sys

sys.path.insert(0, os.path.abspath('../hippomaps'))

try:
    import hippomaps.fetcher
    print("hippomaps.fetcher imported successfully")
except ImportError as e:
    print(f"Error importing hippomaps.fetcher: {e}")

# -- Project information

project = 'Hippomaps'
copyright = '2024, DeKraker'
author = 'DeKraker'

release = '0.1'
version = '0.1.0'

# -- General configuration

extensions = [
    'nbsphinx',
    'sphinx_gallery.load_style',
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosectionlabel',
    'sphinx_gallery.gen_gallery',
    'myst_parser'
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

# source_suffix = '.rst'
source_suffix = ['.rst', '.md']

sphinx_gallery_conf = {
    'examples_dirs': '../../tutorials',
    # 'plot_gallery': 'False',
    'thumbnail_size': (250, 250),
    # 'image_scrapers': ('matplotlib', _get_sg_image_scraper()),
    # 'within_subsection_order': FileNameSortKey,
    'download_all_examples': False,
    'remove_config_comments': True,
    # 'run_stale_examples': True,
}
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

#generate autosummary even if no references
autosummary_generate = True
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

html_theme = 'sphinx_rtd_theme'

# -- Options for EPUB output
epub_show_urls = 'footnote'
html_show_sourcelink = False
html_copy_source = False
