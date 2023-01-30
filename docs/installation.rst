Installation
============

First, clone the repository and install the package:

.. code-block:: bash

   git clone git@github.com:RolandBacon/pyetc/pyetc.git
   cd pyetc
   pip install .
   
Then run the unit tests

.. code-block:: bash

   cd pyetc/tests
   pytest 

The following dependencies are required:

   - python 3.6 or later
   - mpdaf, MUSE Python data analysis framework utilities
   - spextra, a python library for managing and manipulating astronomical spectra
   - numpy
   - astropy


.. warning::

   You will need the latest development version of MPDAF, which can obtained at https://github.com/musevlt/mpdaf


    
    
