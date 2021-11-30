# Bayesian
interpret CPT with the Bayesian approach

This code makes use of several python packages that are relatively hard to install. Thefore, here a quick overview on how to start working with the code!
quick note: This manual uses anaconda. If not already, please install anaconda from: anaconda.org

# STEP 1: SETTING UP AN ENVIRONMENT
1. open the anaconda prompt
2. make a new environment, type:
  conda create -n geo_env
3. activate environment, type:
  conda activate geo_env
4. now you are inside the environment. the next steps will use this environment to install the required packages

# STEP 2: DOWNLOAD GEOPANDAS
1. in your environment type the following:
  conda config --env --add channels conda-forge
  conda config --env --set channel_priority strict
  conda install python=3 geopandas
2. to install geopandas, the program automatically downloads the requirements. This may take several minutes to complete

# STEP 3: DOWNLOAD DESCARTES
1. this package is used to visualize polygons with python. it is installed within the environment using:
  conda install -c conda-forge descartes
  

# STEP 4: DOWNLOAD THE PYTHON FILES 
1. 
1. add the CPT .gef file to the folder
1. main.py contains a list of input parameters. Please fill these in accordingly.

