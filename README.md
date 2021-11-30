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
  

# STEP 4: DOWNLOAD and THE PYTHON FILES 
1. Download and save the files in a directory folder
2. in that same directory, save the .gef file of the CPT that needs to be analyzed
3. in main.py specify the input parameters

#EXAMPLE
1. the current code analyses the CPT000000097976_IMBRO_A.gef file. if all files are in the same directory folder, run main.py to check if the code works!

