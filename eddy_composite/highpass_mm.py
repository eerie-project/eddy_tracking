#!/bin/python
# ©2023 MPI-M, Dian Putrasahan

import sys
varname=sys.argv[1]
yyyy=sys.argv[2]
mth=sys.argv[3]
#varname='zos'
#varname='Wind_Speed_10m'

print('High pass filter monthly '+varname+' for year='+str(yyyy)+', mth='+str(mth))

from py_eddy_tracker.dataset.grid import RegularGridDataset
from datetime import datetime

expid='ngc2013'
fq='mm'
outdir='/work/mh0287/m300466/topaz/'+expid+'/'
date = datetime(int(yyyy), int(mth), 1)

lon_name='lon'
lat_name='lat'
if varname=='to':
    zidx=1
    grid_name=outdir+varname+'/'+fq+'/'+expid+'_'+varname+'_'+str(zidx)+'_'+fq+'_'+date.strftime('%Y%m')+'_MR25.nc'
else:
    grid_name=outdir+varname+'/'+fq+'/'+expid+'_'+varname+'_'+fq+'_'+date.strftime('%Y%m')+'_MR25.nc'

wavelength = 700  #choice of spatial cutoff for high pass filter in km

g = RegularGridDataset(grid_name, lon_name, lat_name) #assumes no time index, might only take the first time index
g.bessel_high_filter(varname, wavelength, order=1) #perfroms only on 1 time index
if varname=='to':
    zidx=1
    g.write(outdir+varname+'/'+fq+'/'+expid+'_'+varname+'_'+str(zidx)+'_'+fq+'_'+date.strftime('%Y%m')+'_MR25_hp'+str(wavelength)+'.nc')
else:
    g.write(outdir+varname+'/'+fq+'/'+expid+'_'+varname+'_'+fq+'_'+date.strftime('%Y%m')+'_MR25_hp'+str(wavelength)+'.nc')


