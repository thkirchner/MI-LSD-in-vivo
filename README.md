# MI-LSD-in-vivo
This is code for the paper "Machine learning enabled multiple illumination quantitative optoacoustic oximetry imaging in humans"

## General notes
Always set the correct path in each .ipynb or .py script where "/SET_PATH/" or similar is specified. The paper data is available at [![DOI](https://zenodo.org/badge/DOI/TODO.svg)](https://doi.org/TODO)

## Reproducing results and figures
``preprocessing.ipynb``
Generates B-Mode images from the raw and unsorted rf data in the in vivo data set.

``generate_gbm_results.ipynb``
Takes these B-Mode images and produces all results including figures. Follow the comments for more information!

``absorption_figure.ipynb``
Reproduces figure 2.
## MCX simulations
Warning: full resimulation will take very long (around 100 days) to compute on a single SOA GPU. Better implement this on HPC. The simulations also use the ippai libary for data management. The exact version of this library is available from the corresponding author, though it is a better idea to build off the updated and renamed [SIMPA](https://github.com/CAMI-DKFZ/simpa) toolkit if you plan to do your own simulations. 

The in silico data sets volumes were simulated with ``python sO2_melanin_sim_array.py VOLUME_ID`` on a HPC cluster. The extracted spectra were generated from the simulated volumes using ``generate_descriptors_TRAIN.py`` and ``generate_descriptors_TEST.py``.
