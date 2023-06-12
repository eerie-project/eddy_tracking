"""
Need to activate conda environment in which py-eddy-tracker is installed:
MO system:
  conda activate /home/h06/hadom/conda/envs/pip-env
JASMIN:
  source /home/users/mjrobert/miniconda3/bin/activate
  conda activate /home/users/mjrobert/miniconda3/envs/pyeddytracker1

Correspondances
===============

Correspondances is a mechanism to intend to continue tracking with new detection

"""

import logging
import glob, os, sys

# %%
from matplotlib import pyplot as plt
from netCDF4 import Dataset

from py_eddy_tracker import start_logger
from py_eddy_tracker.data import get_remote_demo_sample
from py_eddy_tracker.featured_tracking.area_tracker import AreaTracker

# In order to hide some warning
import py_eddy_tracker.observations.observation
from py_eddy_tracker.tracking import Correspondances

py_eddy_tracker.observations.observation._display_check_warning = False

def write_obs_files(c, raw, output_dir, zarr, eddy_type, nb_obs_min):
    kw_write = dict(path=output_dir, zarr_flag=zarr, sign_type=eddy_type)

    fout = os.path.join(output_dir, eddy_type+'_untracked.nc')
    c.get_unused_data(raw_data=raw).write_file(
        filename=fout
    )

    short_c = c._copy()
    short_c.shorter_than(size_max=nb_obs_min)
    short_track = short_c.merge(raw_data=raw)

    if c.longer_than(size_min=nb_obs_min) is False:
        long_track = short_track.empty_dataset()
    else:
        long_track = c.merge(raw_data=raw)

    # We flag obs
    if c.virtual:
        long_track["virtual"][:] = long_track["time"] == 0
        long_track.normalize_longitude()
        long_track.filled_by_interpolation(long_track["virtual"] == 1)
        short_track["virtual"][:] = short_track["time"] == 0
        short_track.normalize_longitude()
        short_track.filled_by_interpolation(short_track["virtual"] == 1)

    print("Longer track saved have %d obs", c.nb_obs_by_tracks.max())
    print(
        "The mean length is %d observations for long track",
        c.nb_obs_by_tracks.mean(),
    )

    fout = os.path.join(output_dir, eddy_type+'_tracks.nc')
    long_track.write_file(filename=fout)
    fout = os.path.join(output_dir, eddy_type+'_short.nc')
    short_track.write_file(
        #filename="%(path)s/%(sign_type)s_track_too_short.nc", **kw_write
        filename=fout
    )

def tracking(file_objects, previous_correspondance, eddy_type, zarr=False, nb_obs_min=10, raw=True, cmin=0.05, virtual=4):
    # %%
    # We run a tracking with a tracker which uses contour overlap, on first time step
    output_dir = os.path.dirname(previous_correspondance)
    class_kw = dict(cmin=cmin)
    if not os.path.exists(previous_correspondance):
        c = Correspondances(
            datasets=file_objects, class_method=AreaTracker, 
            class_kw=class_kw, virtual=virtual
        )
        c.track()
        c.prepare_merging()
    else:
        c = Correspondances(
            datasets=file_objects, class_method=AreaTracker, 
            class_kw=class_kw, virtual=virtual,
            previous_correspondance=previous_correspondance
        )
        c.track()
        c.prepare_merging()
        c.merge()

    new_correspondance = previous_correspondance[:-3]+'_new.nc'
    with Dataset(new_correspondance, "w") as h:
        c.to_netcdf(h)

    try:
        # test can read new file, and then move to replace old file
        nc = Dataset(new_correspondance, 'r')
        os.rename(new_correspondance, previous_correspondance)
    except:
        raise Exception('Error opening new correspondance file '+new_correspondance)

    write_obs_files(c, raw, output_dir, zarr, eddy_type, nb_obs_min)

def get_environment_variables():
    """
    Get the required environment variables from the suite. A list and
    explanation of the required environment variables is included in the
    documentation.
    """
    global um_suiteid, dir_out

    try:
        um_suiteid = os.environ["SUITEID_OVERRIDE"]
    except:
        um_suiteid = os.environ["CYLC_SUITE_NAME"]
    dir_out = os.environ["DIR_OUT"]

def main(eddy_type, zarr, nb_obs_min, raw, cmin):
    get_environment_variables()
    tracker_dir = os.path.join(dir_out, um_suiteid, 'Tracker')
    if not os.path.exists(tracker_dir):
        os.makedirs(tracker_dir)
    search = os.path.join(dir_out, um_suiteid, eddy_type+'_*.nc')
    print('search files ',search)
    file_objects = sorted(glob.glob(search))
    print('file objects ',file_objects)
    previous_correspondance = os.path.join(tracker_dir, eddy_type+'_correspondance.nc')
    tracking(file_objects, previous_correspondance, eddy_type, zarr=zarr, nb_obs_min=nb_obs_min, raw=raw, cmin=cmin)

if __name__ == "__main__":
    eddy_type = sys.argv[1]
    # default options for tracking
    zarr = False # not used currently, but could be to produce zarr format files
    nb_obs_min = 10 # minimum of 10 points in track to be considered a long trajectory
    raw = True # 
    cmin = 0.05 # minimum contour
    virtual = 4 # number of consecutive timesteps with missing detection allowed
    main(eddy_type, zarr, nb_obs_min, raw, cmin, virtual)
