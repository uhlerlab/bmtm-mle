#!/usr/bin/env bash

python3 -m venv venv  # create a virtual environment called `venv`
source venv/bin/activate  # activate the virtual environment
pip3 install numpy scipy sklearn pandas  # basic data manipulation
pip3 install matplotlib seaborn  # plotting
pip3 install tqdm  # progress bars
pip3 install line_profiler  # inspecting performance
pip3 install ipython ipdb  # interactive testing
pip3 install jupyter ipykernel  # for Jupyter notebooks
python3 -m ipykernel install --user --name=project-template
