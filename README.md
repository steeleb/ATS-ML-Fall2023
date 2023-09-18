# ATS-ML-Fall2023

Repository of code for ATS780A7 assignments

To run code in this repository, it requires a virtual environment (here named
'env_ATSML'). To create a virtual environment for this codebase, run the 
following commands at the terminal. The following venv setup is courtesy of 
Dr. Elizabeth Barnes, Colorado State University.

- conda create --name env_ATSML python=3.10.10
- conda activate env_ATSML
- conda install -c apple -c conda-forge -c nodefaults tensorflow-deps
- python -m pip install tensorflow-macos==2.10.0
- python -m pip install tensorflow-metal==0.6.0
- pip install tensorflow-probability==0.15 silence-tensorflow
- conda install numpy scipy matplotlib scikit-learn jupyterlab
- pip install pandas statsmodels icecream palettable seaborn progressbar2 tabulate isort
- pip install tqdm pydot graphviz
- pip install -U scikit-image
- pip install shap
- conda install pandas
