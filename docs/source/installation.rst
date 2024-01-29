Installation
=================


Setting Python environment
------------

To use HippoMaps, we recommend creating a separate python environment first:

.. code-block:: console

   $ virtualenv -p python3.9 venv
   $ source venv/bin/activate

Install
------------

To install HippoMaps, clone and `pip` the repository:

.. code-block:: console

   $ git clone https://github.com/HippAI/hippomaps.git
   $ pip install -e hippomaps

Test
------------

You should now be able to access HippUnfold from your python console. For example:

>>> import hippomaps as hm
