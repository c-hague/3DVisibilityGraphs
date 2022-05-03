# 3DVisibilityGraphs
Project for ITCS 8151 at University of North Carolina at Charlotte. Finds the Dubins airplane path between two configuration while avoiding obstacles.
## Installation 
[Anaconda]("https://www.anaconda.com/") is used to install all required dependencies.
Install dependencies with ``conda env create -f environment.yaml``.
To run application first activate conda environment ``conda activate env``.
To run application ``python -m visibilitygraphs --help``.
## Visibility graph experiment
``python -m visibilitygraphs -150 -325 15 0 0 ./data/uptownCharlotte.obj -r 40 -i -92 303 15 4.71 0 -f .707 -t 1 --levels 10 -p true --distance 50``
## RRT experiment
``python -m visibilitygraphs -150 -325 15 0 0 ./data/uptownCharlotte.obj -r 5 -i -92 303 15 0 0 -f .707 -t 2 --points 2000 --distance 20 --rgoal 20 -p true``