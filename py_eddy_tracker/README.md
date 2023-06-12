This code uses py-eddy-tracker
https://py-eddy-tracker.readthedocs.io/en/latest/index.html
and is based on discussion with developer about how to put this into a workflow as new data is available, 
using values as in the AVISO eddy track datasets
https://github.com/AntSimi/py-eddy-tracker/discussions/198

In workflow (at Met Office or JASMIN), the code takes in daily ocean data on 0.25degree regular grid with variable name - zos as default
(or produces it as part of pre-processing, not included here yet)
eddy_identify.py identifies the eddies in each daily data file
eddy_tracker.py does incremental tracking using the outputs of eddy_identify.py
