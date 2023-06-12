"""
Eddy detection on SLA and ADT
=============================
Need to activate conda environment in which py-eddy-tracker is installed:
MO system:
  conda activate /home/h06/hadom/conda/envs/pip-env
JASMIN:
  source /home/users/mjrobert/miniconda3/bin/activate
  conda activate /home/users/mjrobert/miniconda3/envs/pyeddytracker1

Identify with code here (or on command line, see https://py-eddy-tracker.readthedocs.io/en/stable/grid_identification.html)

"""
from datetime import datetime, timedelta
import cftime

#from matplotlib import pyplot as plt

from py_eddy_tracker import data
from py_eddy_tracker.dataset.grid import RegularGridDataset
from netCDF4 import Dataset
import os, glob, sys
import time

def get_data_and_filter(fname, var, wavelength, wavelength_order):
    g = RegularGridDataset(
        fname,
        "lon",
        "lat",
    )
    #g.add_uv("adt", "ugos", "vgos")
    g.add_uv(var, "ugos", "vgos")
    #g.copy("adt", "adt_raw")
    g.copy(var, "sla_raw")
    #g.bessel_high_filter("adt", wavelength)
    g.bessel_high_filter(var, wavelength, order=wavelength_order)

    return g

def eddy_ident(g, date, var, isoline_step, shape_error, sampling):

    #isoline_step = 0.002
    #pixel_limit=(5, 2000), # Min and max pixel count for valid contour
    #shape_error=55, # Error max (%) between ratio of circle fit and contour

    #a_sla, c_sla = g.eddy_identification(var, 'ugos', 'vgos', date, isoline_step, pixel_limit=pixel_limit, shape_error=shape_error)
    a_sla, c_sla = g.eddy_identification(var, 'ugos', 'vgos', date, isoline_step, shape_error=shape_error, sampling=sampling)

    return a_sla, c_sla

def extract_data(fname, var, index):
    time_out = str(index).zfill(2)
    date = datetime(2014, 1, index+1)
    fout = fname[:-3]+'_'+time_out+'.nc'
    return fout, date

def get_environment_variables():
    """
    Get the required environment variables from the suite. A list and
    explanation of the required environment variables is included in the
    documentation.
    """
    global um_runid, um_suiteid, cylc_task_cycle_time, time_cycle, previous_cycle, tm2_cycle, next_cycle, startdate, enddate, current_year, current_month, current_day, period, cycleperiod, dir_out, calendar, variable

    try:
        um_suiteid = os.environ["SUITEID_OVERRIDE"]
    except:
        um_suiteid = os.environ["CYLC_SUITE_NAME"]
    um_runid = um_suiteid.split('-')[1]
    cylc_task_cycle_time = os.environ["CYLC_TASK_CYCLE_TIME"]
    time_cycle = os.environ["TIME_CYCLE"]
    startdate = os.environ["STARTDATE"]
    enddate = os.environ["ENDDATE"]
    cycleperiod = os.environ["CYCLEPERIOD"]
    dir_out = os.environ["DIR_OUT"]
    calendar = os.environ["CALENDAR"]
    variable = os.environ["VAR_SSH"]

    current_year = time_cycle[0:4]
    current_month = time_cycle[4:6]
    current_day = time_cycle[6:8]
    period = str(current_year)+str(current_month)+str(current_day)

def make_tracker_input_files(fname_ac, fname_c, max_missing=3, long_duration_min=10):
    '''
    Make the .yaml files to feed into the tracker when using 
    the non-python tracking method
    '''
    dir_name = os.path.dirname(fname_ac)
    dir_save = os.path.join(os.path.dirname(fname_ac), 'Tracker')
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
    
    tracker_anticyclonic_yaml = os.path.join(dir_save, 'tracker_anticyclonic.yaml')
    tracker_cyclonic_yaml = os.path.join(dir_save, 'tracker_cyclonic.yaml')

    for fout, eddy_type in zip([tracker_anticyclonic_yaml, tracker_cyclonic_yaml], ['Anticyclonic', 'Cyclonic']):
        with open(fout, 'w') as fo:
            fo.write('PATHS: \n')
            fo.write('  # Files from eddy identification \n')
            value = os.path.join(dir_name, eddy_type+'*.nc')
            fo.write('  FILES_PATTERN: '+value+'\n')
            fo.write('  SAVE_DIR: '+dir_save+'\n')
            fo.write('# Number of consecutive timesteps with missing detection allowed'+'\n')
            fo.write('VIRTUAL_LENGTH_MAX: '+str(max_missing)+'\n')
            fo.write('# Minimal number of timesteps to considered as a long trajectory'+'\n')
            fo.write('TRACK_DURATION_MIN: '+str(long_duration_min)+'\n')

def main(wavelength, wavelength_order, isoline_step, shape_error, sampling, calendar, variable='zos'):

    get_environment_variables()
    eddy_detect_dir = os.path.join(dir_out, um_suiteid)
    if 'M' in cycleperiod:
        dates = current_year+'-'+current_month+'-*.nc'
    elif 'Y' in cycleperiod:
        dates = current_year+'-??-??.nc'
    elif 'D' in cycleperiod:
        dates = current_year+'-'+current_month+'-'+current_day[0]+'*.nc'

    search = os.path.join(eddy_detect_dir, variable+'_'+dates)
    fnames = sorted(glob.glob(search))
    if len(fnames) == 0:
        print('no files to identify found ',search)
        raise Exception('No files to identify')

    for nf, f in enumerate(fnames):
    #for nf, f in enumerate(fnames[-3:]):
        date_str = os.path.basename(f).split('_')[1][0:10]
        print('date ',date_str)
        #date = cftime.datetime(int(date_str[0:4]), int(date_str[5:7]), int(date_str[8:]), calendar = calendar)
        year = int(date_str[0:4])
        month = int(date_str[5:7])
        day = int(date_str[8:10])
        # use cftime for 360_day calendar (which is not supported by default
        # by the eddy_identification, so I added a case there for cftime)
        if '360' in str(calendar):
            date = cftime.datetime(year, month, day, calendar=calendar)
        else:
            date = datetime(year, month, day)
        date_str1 = date.strftime('%Y%m%d.nc')
        fname_ac = os.path.join(eddy_detect_dir, 'Anticyclonic_'+date_str1)
        fname_c = os.path.join(eddy_detect_dir, 'Cyclonic_'+date_str1)
        if not os.path.exists(fname_ac) or not os.path.exists(fname_c):
            print(date)
            g = get_data_and_filter(f, variable, wavelength, wavelength_order)
            a_sla, c_sla = eddy_ident(g, date, variable, isoline_step, shape_error, sampling)
            print('g ',g)
            with Dataset(fname_ac, 'w') as h_a:
                a_sla.to_netcdf(h_a)
            with Dataset(fname_c, 'w') as h_c:
                c_sla.to_netcdf(h_c)
        #if nf == 0:
        #    make_tracker_input_files(fname_ac, fname_c)

        # test that the netcdf files are completed and not corrupt
        #time.sleep(5)
        try:
            ncid = Dataset(fname_ac)
        except:
            os.remove(fname_ac)
            raise Exception('Corrupt netcdf file '+fname_ac)
        try:
            ncid = Dataset(fname_c)
        except:
            os.remove(fname_ac)
            raise Exception('Corrupt netcdf file '+fname_c)

if __name__ == "__main__":

    # %%
    # Default values (as used in AVISO standard identify/tracking)
    wavelength = 700
    wavelength_order = 1
    isoline_step = 0.002
    shape_error = 70 # Error max (%) between ratio of circle fit and contour
    sampling = 20 # accuracy of storing eddy data
    calendar = '360_day' # or gregorian
    variable = 'zos' # variable name in the input netcdf files

    main(wavelength, wavelength_order, isoline_step, shape_error, sampling, calendar, variable=variable)

