#!/usr/bin/env bash
conda create -n py2_django python=2 anaconda
source activate py2_django
pip install django gensim prettytable
pip install numpy==1.12.1



