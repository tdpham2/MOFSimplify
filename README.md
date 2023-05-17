Forked MOFSimplify from https://github.com/hjkgrp/MOFSimplify

This version modified app.py functions to run thermal stability prediction for many CIFs at once.

Tested with the following dependencies: python=3.9, pymatgen=2023.3.23, keras=2.12.0, scikit-learn=1.2.2, pandas=1.5.3, tensorflow=2.12.0, molsimplify=1.7.2

TODO: Create function to do run_solvent_ANN() for multiple CIFs

Usage:
- The script to run thermal_ANN is start_batch.py. Specify the cif_folder parameters, and run the script as python start_batch.py. 
