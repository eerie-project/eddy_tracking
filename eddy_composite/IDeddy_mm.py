#!/bin/python

# ©2023 MPI-M, Dian Putrasahan

import sys
yyyy=sys.argv[1]
mth=sys.argv[2]

print('Identifying monthly eddies for year='+str(yyyy)+', mth='+str(mth))

from py_eddy_tracker.dataset.grid import RegularGridDataset
from datetime import datetime

#Read in example SSH data that has been mapped onto a 0.25deg regular grid.
# /work/mh0287/m300466/topaz/ngc2013/zos
# /work/mh0287/m300466/topaz/rthk001/zos

expid='ngc2013'
varname='zos'
fq='mm'
outdir='/work/mh0287/m300466/topaz/'+expid+'/'+varname+'/'+fq+'/'
date = datetime(int(yyyy), int(mth), 1)
grid_name, lon_name, lat_name = (
    outdir+expid+'_'+varname+'_'+fq+'_'+date.strftime('%Y%m')+'_MR25.nc',
    "lon",
    "lat",
)
g = RegularGridDataset(grid_name, lon_name, lat_name) #assumes no time index, might only take the first time index

from py_eddy_tracker import data
from py_eddy_tracker.dataset.grid import RegularGridDataset

# from py_eddy_tracker import start_logger
# start_logger().setLevel("DEBUG")  # Available options: ERROR, WARNING, INFO, DEBUG

##Load Input grid, SSH is used to detect eddies. Add a new filed to store the high-pass filtered SSHA
g.add_uv("zos")
g.copy("zos", "zos_high")
wavelength = 700  #choice of spatial cutoff for high pass filter in km
g.bessel_high_filter("zos", wavelength, order=1)

# Run the detection for the total grid and the filtered grid
step_ht=0.005 #intervals to search for closed contours (5mm in this case)
a, c = g.eddy_identification(
    "zos_high", "u", "v",
    date,  # Date of identification
    step_ht,  # step between two isolines of detection (m)
    pixel_limit=(50, 400),  # Min and max pixel count for valid contour
    shape_error=30,  # Error max (%) between ratio of circle fit and contour
)

#Save output
from netCDF4 import Dataset

with Dataset(date.strftime(outdir+expid+"_anticyclonic_"+fq+"_"+date.strftime('%Y%m')+".nc"), "w") as h:
    a.to_netcdf(h)
with Dataset(date.strftime(outdir+expid+"_cyclonic_"+fq+"_"+date.strftime('%Y%m')+".nc"), "w") as h:
    c.to_netcdf(h)



