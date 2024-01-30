'''
Code to plot eddy properties from tracks generated with pyeddytracker, but 
not using code from that package (because it takes too much memory/time to 
read via pyeddytracker's utilities
'''

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import numpy as np
import os
import pickle
import iris
import iris.cube
import iris.coord_systems
import cf_units

# temporary directory for intermediate files
eddy_prop_dir = '/scratch/hadom/eddy_prop_dir'
if not os.path.exists(eddy_prop_dir):
    os.makedirs(eddy_prop_dir)
fig_dir = '/home/h06/hadom/workspace/tenten/variability/eddy/figs'

eddy_types = ['cyclonic', 'anticyclonic']

# minimum lifetime (days) of eddies used to plot tracks
min_lifetime_plot_tracks = 365
# minimum lifetime (days) of eddies used for property pdfs etc
min_lifetime_pdfs = 10

# number of years of data to use
nyears = 5

### This section is for Met Office models, which use "suite" names to identify simulations
suites = ['u-cx993', 'aviso']
# start year for each model/obs data
years_start = {}
years_start['u-cx993'] = 1851
years_start['aviso'] = 1993
model_names = ['MOHC_N640O12', 'AVISO']

# paths to files with eddy tracks
file_all = {}
file_all['anticyclonic'] = "/scratch/hadom/eddy_tracking_CMOR/{}/Tracker/Anticyclonic_tracks.nc"
file_all['cyclonic'] = "/scratch/hadom/eddy_tracking_CMOR/{}/Tracker/Cyclonic_tracks.nc"

# obs data from https://www.aviso.altimetry.fr/en/data/products/value-added-products/global-mesoscale-eddy-trajectory-product.html
file_obs = {}
file_obs['anticyclonic'] = "/data/users/hadom/AVISO/META3.2_DT_allsat_Anticyclonic_long_19930101_20220209.nc"
file_obs['cyclonic'] = "/data/users/hadom/AVISO/META3.2_DT_allsat_Cyclonic_long_19930101_20220209.nc"

# colours used for histograms
colours = {}
colours['cyclonic'] = ['cyan', 'lightblue']
colours['anticyclonic'] = ['orange', 'magenta']

# eddy properties and the associated factors, bins for the pdfs
eddy_prop_1 = ['speed_radius', 'speed_average', 'amplitude', 'effective_contour_shape_error']
eddy_prop_1_factor = [0.001, 100, 100, 1]
eddy_prop_1_bins = [np.arange(0, 2000, 1), np.arange(0, 1000, 0.5), np.arange(0.0005, 1000, 0.2), np.arange(0, 100, 1)]

# all of the eddy properties in the netcdf file (for future reference)
#eddy_props = ['effective_area', 'effective_contour_height', 'effective_contour_shape_error', 'effective_radius', 'inner_contour_height', 'speed_area', 'speed_contour_height', 'speed_contour_shape_error']

def load_data(fname, year_end, suite, min_lifetime=None):
    # Load dataset with xarray
    print('load file ',fname)
    dstracks = xr.open_dataset(fname)
    print('times ',suite, dstracks["time.year"])

    # Bounds of the area
    lon_min, lon_max, lat_min, lat_max = 0, 360, -70, 70

    subset = dstracks.sel(obs=(dstracks['time.year'].values <= year_end))

    if min_lifetime is not None:
        subset_lon_life = subset.sel(obs=subset.observation_number>min_lifetime)
        subset = subset.isel(obs=np.in1d(subset.track, subset_lon_life.track))

    print('subset ',subset)
    print('subset ',subset['time.year'].values)

    return subset

def read_datafiles(suite, eddy_t, nyears, min_lifetime):
    eddy_subset = {}
    year_end = years_start[suite] + nyears - 1
    if suite == 'aviso':
        file_in = file_obs
    else:
        file_in = file_all

    pickle_file = os.path.join(os.path.dirname(file_in[eddy_t].format(suite)), suite+'_'+'_'+eddy_t+'_'+str(years_start[suite])+'-'+str(year_end)+'_'+str(min_lifetime)+'minlifetime.pkl')
    if not os.path.exists(pickle_file):
        subset = load_data(file_in[eddy_t].format(suite), year_end, suite, min_lifetime=min_lifetime)
        eddy_subset[(suite, eddy_t)] = subset
        with open(pickle_file, 'wb') as fh:
            pickle.dump(subset, fh)
    else:
        print('open pickle file ',pickle_file)
        with open(pickle_file, 'rb') as fh:
            subset = pickle.load(fh)
            eddy_subset[(suite, eddy_t)] = subset

    return eddy_subset

def plot_tracks_per_model(subset, ntracks, eddy_type, fig, ax, proj, nfig, nyears, suite, isu, min_lifetime):
    if eddy_type == 'cyclonic':
        colour = 'blue'
    else:
        colour = 'red'
    # Plot selected data
    ax.scatter(
        subset.longitude,
        subset.latitude,
        #c=subset.track,
        c=colour,
        s=0.1,
        transform=proj,
        linewidth=0.,
        cmap='tab20b',
        rasterized=True)

    # Add title
    if nfig == 0:
        ax.set_title(" Trajectories longer than "+str(min_lifetime)+" days over "+str(nyears)+' years, '+model_names[isu], fontsize=10)
    if eddy_type == 'cyclonic':
        ax.text(60, 55, 'C: '+str(ntracks), c = colour)
    else:
        ax.text(60, 45, 'A: '+str(ntracks), c = colour)
    # Active meridian/parallel
    if nfig == 0:
        ax.gridlines(draw_labels=True)
    # Active coastline
    ax.coastlines()
    # Display figure
    #plt.savefig('./'+suite+'_'+str(year_max)+'_'+str(min_lifetime)+'.png')

def start_axes(title):
    fig = plt.figure(figsize=(13, 5))
    ax = fig.add_axes([0.03, 0.03, 0.90, 0.94])
    ax.set_xlim(-6, 36.5), ax.set_ylim(30, 46)
    ax.set_aspect("equal")
    ax.set_title(title)
    return ax

def update_axes(ax, mappable=None):
    ax.grid()
    if mappable:
        plt.colorbar(mappable, cax=ax.figure.add_axes([0.95, 0.05, 0.01, 0.9]))

def _cube_data(data):
    """Returns a cube given a list of lat lon information."""
    cube = iris.cube.Cube(data)
    lat_lon_coord_system = iris.coord_systems.GeogCS(6371229.0)
    
    step = 2.0
    start = step/2
    count = 180
    pts = start + np.arange(count, dtype=np.float32) * step
    lon_coord = iris.coords.DimCoord(pts, standard_name='longitude', units='degrees', 
                                 coord_system = lat_lon_coord_system, circular=True)
    lon_coord.guess_bounds()
    
    start = -90
    step = 2.0
    count = 90
    pts = start + np.arange(count, dtype=np.float32) * step
    lat_coord = iris.coords.DimCoord(pts, standard_name='latitude', units='degrees', 
                                 coord_system = lat_lon_coord_system)
    lat_coord.guess_bounds()
    
    cube.add_dim_coord(lat_coord, 0)
    cube.add_dim_coord(lon_coord, 1)
    return cube

def _binned_cube(lats, lons):
    """ Returns a cube (or 2D histogram) of lat/lons locations. """   
    data = np.zeros(shape=(90,180))
    binned_cube = _cube_data(data)
    xs, ys = binned_cube.coord('longitude').contiguous_bounds(), binned_cube.coord('latitude').contiguous_bounds()
    binned_data, _, _ = np.histogram2d(lons, lats, bins=[xs, ys])
    binned_cube.data = np.transpose(binned_data)
    #binned_cube.attributes.pop('history', None) 
    binned_cube.units = cf_units.Unit(1)
    return binned_cube

def flatten(xss):
    return [x for xs in xss for x in xs]

def storm_lats_lons(eddy_subset, plot_type='genesis'):
    """ 
    Returns array of latitude and longitude values for all storms that 
    occurred within a desired year, month set and basin. 
    
    To get genesis, lysis or max intensity results set:
    Genesis plot: genesis=True
    Lysis plot: lysis=True
    Maximum intensity (location of max wind): 
    max_intensity=True
    
    """
    lats, lons = [], []
    count = 0
    if plot_type == 'genesis':
        first_points = np.argwhere(eddy_subset.observation_number.values == 1)
        lats_np = eddy_subset.latitude.values[first_points]
        lons_np = eddy_subset.longitude.values[first_points]
        lats_flat = flatten(lats_np.tolist())
        lons_flat = flatten(lons_np.tolist())
        lats.extend(lats_flat)
        lons.extend(lons_flat)
    #elif lysis:
    #    lats.extend([storm.obs_at_lysis().lat])
    #    lons.extend([storm.obs_at_lysis().lon])
    #elif max_intensity:
    #    lats.extend([storm.obs_at_vmax().lat])
    #    lons.extend([storm.obs_at_vmax().lon])
    #else:
    elif plot_type == 'density':
        lats_np = eddy_subset.latitude.values.tolist()
        lons_np = eddy_subset.longitude.values.tolist()
        #lats_flat = flatten(lats_np.tolist())
        #lons_flat = flatten(lons_np.tolist())
        lats.extend(lats_np)
        lons.extend(lons_np)
    count += 1
                
    # Normalise lon values into the range 0-360
    norm_lons = []
    for lon in lons:
        norm_lons.append((lon + 720) % 360)
    return lats, norm_lons, count

def calc_birth(suites, min_lifetime, plot_type='genesis', nyears=5):
    '''
    For genesis, need to get the position of the first point of each eddy, and then the lat/lon of that
    For density, need all the lat/lon of each track
    '''

    eddy_genesis = {}
    for isu, suite in enumerate(suites):
        for eddy_type in eddy_types:
            nc_file = os.path.join(eddy_prop_dir, suite+'_'+eddy_type+'_'+plot_type+'_'+str(nyears)+'years_'+str(min_lifetime)+'minlifetime.nc')
            if not os.path.exists(nc_file):
                eddy_subset = read_datafiles(suite, eddy_type, nyears, min_lifetime)
                eddy = eddy_subset[(suite, eddy_type)]
                lats, lons, count = storm_lats_lons(eddy, plot_type=plot_type)
                cube = _binned_cube(lats, lons)
                cube /= nyears
                print('Total number of storms in time period:', suite, eddy_type, nyears, count)
                eddy_genesis[(suite, eddy_type)] = cube
                iris.save(cube, nc_file)

def plot_birth(suites, min_lifetime, plot_type='density', nyears=5):

    fig = plt.figure(figsize=(13,7),dpi=100)
    proj=ccrs.PlateCarree()
    for ie, eddy_type in enumerate(eddy_types):
        for isu, suite in enumerate(suites):
            nc_file = os.path.join(eddy_prop_dir, suite+'_'+eddy_type+'_'+plot_type+'_'+str(nyears)+'years_'+str(min_lifetime)+'minlifetime.nc')
            cube = iris.load_cube(nc_file)
            # Create subplot
            ax = fig.add_subplot(2,2,ie*2+isu+1, projection=proj)
            m = ax.pcolormesh(cube.coord('longitude').points, cube.coord('latitude').points, cube.data, cmap='terrain_r', vmin=0, vmax=250)
            plt.title(suite+' '+eddy_type)
            cb = plt.colorbar(m, cax=ax.figure.add_axes([0.94, 0.2, 0.01, 0.6]))
    fig.subplots_adjust(bottom=0.04, left=0.05, right=.91, top=.95, wspace=0.06, hspace=0.0)
    plt.suptitle('All eddies '+plot_type+' over '+str(nyears)+' years, '+str(min_lifetime)+' min lifetime')
    #plt.tight_layout()
    suite_names = '_'.join(suites)
    figname = os.path.join(fig_dir, 'eddy_'+plot_type+'_'+suite_names+'_'+str(nyears)+'years_'+str(min_lifetime)+'minlifetime.pdf')
    plt.savefig(figname)
    plt.savefig(figname[:-3]+'png')
    plt.show()

def plot_birth_pyeddytracker(suites, subsets):
    fig = plt.figure(figsize=(10,5),dpi=100)
    proj=ccrs.PlateCarree()
    # Create subplot
    ax = fig.add_subplot(111, projection=proj)

    #t0, t1 = subset.period
    step = 0.125
    bins = ((-10, 37, step), (30, 46, step))
    #kwargs = dict(cmap="terrain_r", factor=100 / (t1 - t0), name="count", vmin=0, vmax=1)
    kwargs = dict(cmap="terrain_r", name="count", vmin=0, vmax=1)

    ax = start_axes("Birth cyclonic frequency (%)")
    suite = suites[0]
    subset = subsets[(suite, 'cyclonic')]
    g_c_first = subset.first_obs().grid_count(bins, intern=True)
    m = g_c_first.display(ax, **kwargs)
    update_axes(ax, m)
    plt.show()

def plot_amplitude_pyeddytracker(suites, subsets):
    step = 0.1
    ax = start_axes("Amplitude mean by box of %s°" % step)
    suite = suites[0]
    a = subsets[(suite, 'cyclonic')]
    print('a ',a)

    bins = ((-7, 37, step), (30, 46, step))
    kwargs = dict(cmap="terrain_r", name="count", vmin=0, vmax=10)
    ax = start_axes("Amplitude (cm)")

    #g = a.grid_stat(((-7, 37, step), (30, 46, step)), "amplitude")
    g_c_first = a['amplitude'].grid_count(bins, intern=True)
    #m = g.display(ax, name="amplitude", vmin=0, vmax=10, factor=100)
    #ax.grid()
    m = g_c_first.display(ax, **kwargs)
    update_axes(ax, m)

    cb = plt.colorbar(m, cax=ax.figure.add_axes([0.94, 0.05, 0.01, 0.9]))
    cb.set_label("Amplitude (cm)")
    plt.show()

def plot_radius_pyeddytracker(suites, subsets):
    step = 0.1
    ax = start_axes("Speed radius mean by box of %s°" % step)
    suite = suites[0]
    a = subsets[(suite, 'cyclonic')]

    g = a.grid_stat(((-7, 37, step), (30, 46, step)), "radius_s")
    m = g.display(ax, name="radius_s", vmin=10, vmax=50, factor=0.001)
    ax.grid()
    cb = plt.colorbar(m, cax=ax.figure.add_axes([0.94, 0.05, 0.01, 0.9]))
    cb.set_label("Speed radius (km)")
    plt.show()

def plot_tracks(suites, nyears, min_lifetime=10):
    # Projection
    proj=ccrs.PlateCarree()
    # Create figure
    nsuites = len(suites)
    if nsuites == 1:
        fig = plt.figure(figsize=(10,5),dpi=100)
        ny = 1; nx = 1
    elif nsuites == 2:
        fig = plt.figure(figsize=(10,10),dpi=100)
        ny = 1; nx = 2
    else:
        fig = plt.figure(figsize=(15,10),dpi=100)
        ny = 2; nx = 2

    for isu, suite in enumerate(suites):
        # Create subplot
        ax = fig.add_subplot(nx, ny, isu+1, projection=proj)

        for ie, eddy_t in enumerate(eddy_types):
            eddy_subset = read_datafiles(suite, eddy_t, nyears, min_lifetime)        
            subset = eddy_subset[(suite, eddy_t)]
            years = subset["time.year"].values
            months = subset["time.month"].values
            nyears = (years[-1] - years[0]) + 1
            print('years, nyears ', suite, years, nyears)

            # Create the final subset
            # print('subset.track ',subset.track)
            unique, counts = np.unique(subset.track, return_counts=True)
            ntracks = len(counts)
            print('No tracks ', ntracks)
            plot_tracks_per_model(subset, ntracks, eddy_t, fig, ax, proj, ie, nyears, suite, isu, min_lifetime)

    fig.subplots_adjust(bottom=0.07, left=0.12, right=.9, top=.9, wspace=0.2, hspace=0.26)
    plt.tight_layout()
    figname = os.path.join(fig_dir, 'eddy_tracks_' + '_'.join(suites) + '_'+str(min_lifetime)+'minlifetime')
    plt.savefig(figname + '.pdf')
    plt.savefig(figname + '.png')
    plt.show()

def plot_lifetime_histogram(a, suites, nyears, min_lifetime):
    nb_year = nyears

    # %%
    # Setup axes
    figure = plt.figure(figsize=(12, 8))
    ax_ratio_cum = figure.add_axes([0.55, 0.06, 0.42, 0.34])
    ax_ratio = figure.add_axes([0.07, 0.06, 0.46, 0.34])
    ax_cum = figure.add_axes([0.55, 0.43, 0.42, 0.54])
    ax = figure.add_axes([0.07, 0.43, 0.46, 0.54])
    ax.set_ylabel("Eddies by year")
    ax_ratio.set_ylabel("Ratio Cyclonic/Anticyclonic")
    for ax_ in (ax, ax_cum, ax_ratio_cum, ax_ratio):
        ax_.set_xlim(0, 730)
        if ax_ in (ax, ax_cum):
            ax_.set_ylim(1e-1, 5e4), ax_.set_yscale("log")
        else:
            ax_.set_xlabel("Lifetime in days (by week bins)")
            ax_.set_ylim(0, 2)
            ax_.axhline(1, color="black", lw=2)
        ax_.grid()
    ax_cum.xaxis.set_ticklabels([]), ax_cum.yaxis.set_ticklabels([])
    ax.xaxis.set_ticklabels([]), ax_ratio_cum.yaxis.set_ticklabels([])
    ax_ratio.set_title('Histogram', fontsize='medium')
    ax_ratio_cum.set_title('Cumulative histogram', fontsize='medium')

    # plot data
    bin_hist = np.arange(7, 2000, 7)
    x = (bin_hist[1:] + bin_hist[:-1]) / 2.0
    cum = {}
    nb = {}
    eddy_lifehist = {}
    eddy_cum = {}
    for suite in suites:
        for eddy_type in ['cyclonic', 'anticyclonic']:
            a_nb = a[(suite, eddy_type)][:,1]
            a_nb = a_nb[a_nb != 0]
            w_a = np.ones(a_nb.shape) / nb_year
            if eddy_type == 'cyclonic':
                colour = 'cyan'
                if suite == 'aviso':
                    colour = 'blue'
            else:
                colour = 'orange'
                if suite == 'aviso':
                    colour = 'red'
            eddy_lifehist[(suite, eddy_type)], bin_edges = np.histogram(a_nb, bins=bin_hist, weights=w_a)
            dist = bin_edges[1] - bin_edges[0]
            print('dist ',eddy_type, suite, dist)
            cumsum = np.cumsum(eddy_lifehist[(suite, eddy_type)][::-1])[::-1]
            #eddy_cum[(suite, eddy_type)] = np.subtract(1.0, cumsum)
            eddy_cum[(suite, eddy_type)] = cumsum

            kwargs_a = dict(histtype="step", bins=bin_hist, x=a_nb, color=colour, weights=w_a)
            #cum[eddy_type], _, _ = ax_cum.hist(cumulative=-1, **kwargs_a)
            #nb[eddy_type], _, _ = ax.hist(label=eddy_type+','+suite, **kwargs_a)
            ax_cum.plot(x, eddy_cum[(suite, eddy_type)], color=colour)
            ax.plot(x, eddy_lifehist[(suite, eddy_type)], color=colour, label=eddy_type+','+suite)
        colour = 'blue'
        if suite == 'aviso':
            colour = 'cyan'
        #ax_ratio_cum.plot(x, cum['cyclonic'] / cum['anticyclonic'], color=colour)
        #ax_ratio.plot(x, nb['cyclonic'] / nb['anticyclonic'], color=colour)
        ax_ratio_cum.plot(x, eddy_cum[(suite, 'cyclonic')] / eddy_cum[(suite, 'anticyclonic')], color=colour)
        ax_ratio.plot(x, eddy_lifehist[(suite, 'cyclonic')] / eddy_lifehist[(suite, 'anticyclonic')], color=colour)
        ax.legend()

    suite_names = '_'.join(suites)
    plt.title('Eddy lifetimes for '+suite_names+' over '+str(nyears)+' years with '+str(min_lifetime)+' min lifetime')
    figname = os.path.join(fig_dir, 'eddy_lifetime_'+suite_names+'_'+str(nyears)+'years_'+str(min_lifetime)+'minlifetime.png')
    plt.savefig(figname)

def calculate_eddy_lifetime(eddies):
    total_tracks, count = np.unique(eddies.track, return_counts=True)
    print('total_tracks ',eddies.track)
    track_lengths = np.asarray((total_tracks, count)).T
    return track_lengths

def lifetime_histogram(suites, nyears, min_lifetime):
    eddy_lengths = {}
    for isu, suite in enumerate(suites):
        for eddy_type in eddy_types:
            pickle_file = os.path.join(eddy_prop_dir, suite+'_'+eddy_type+'_lifetime_'+str(nyears)+'years.pkl')
            if not os.path.exists(pickle_file):
                eddy_subset = read_datafiles(suite, eddy_type, nyears, min_lifetime)        
                lifetime = calculate_eddy_lifetime(eddy_subset[suite, eddy_type])
                with open(pickle_file, 'wb') as fh:
                    pickle.dump(lifetime, fh)
            else:
                with open(pickle_file, 'rb') as fh:
                    lifetime = pickle.load(fh)

            eddy_lengths[(suite, eddy_type)] = lifetime
    plot_lifetime_histogram(eddy_lengths, suites, nyears, min_lifetime)
    plt.show()

def calculate_eddy_properties(suite, nyears, min_lifetime):
    for eddy_type in eddy_types:
        eddy_subset = read_datafiles(suite, eddy_type, nyears, min_lifetime)        
        eddy = eddy_subset[(suite, eddy_type)]
        for ina, name in enumerate(eddy_prop_1):
            pickle_file = os.path.join(eddy_prop_dir, suite+'_'+eddy_type+'_'+name+'_'+str(nyears)+'years_'+str(min_lifetime)+'minlifetime_hist.pkl')
            print('process ', pickle_file)
            if not os.path.exists(pickle_file):
                if name == 'speed_radius':
                    eddy_val = eddy.speed_radius.values
                elif name == 'amplitude':
                    eddy_val = eddy.amplitude.values
                elif name == 'speed_average':
                    eddy_val = eddy.speed_average.values
                elif name == 'effective_contour_shape_error':
                    eddy_val = eddy.effective_contour_shape_error.values
            
                factor = eddy_prop_1_factor[ina]
                bins = eddy_prop_1_bins[ina]
                hist, bin_edges = np.histogram(eddy_val * factor, bins=bins, density=True)
                with open(pickle_file, 'wb') as fh:
                    pickle.dump(hist, fh)
                    pickle.dump(bin_edges, fh)

def plot_eddy_properties(suites):
    eddy_hist = {}; eddy_cum = {}
    for isu, suite in enumerate(suites):
        for name in eddy_prop_1:
            for eddy_t in eddy_types:
                pickle_file = os.path.join(eddy_prop_dir, suite+'_'+eddy_t+'_'+name+'_'+str(nyears)+'years_hist.pkl')
                print('read pickle ',pickle_file)
                with open(pickle_file, 'rb') as fh:
                    hist = pickle.load(fh)
                    bin_edges = pickle.load(fh)
                    eddy_hist[(suite, name, eddy_t)] = hist
                dist = bin_edges[1] - bin_edges[0]
                print('dist ',eddy_t, suite, name, dist)
                cumsum = np.cumsum(eddy_hist[(suite, name, eddy_t)]*dist)
                eddy_cum[(suite, name, eddy_t)] = np.subtract(1.0, cumsum)
                print('eddy_cum[eddy_t] ',eddy_cum[(suite, name, eddy_t)])

    fig = plt.figure(figsize=(12, 8))
    kwargs_a = dict(histtype="step", density=True)
    kwargs_c = dict(histtype="step", density=True)
    #hist_max = [0.08, 0.03, 0.4]

    for x0, name, title, xmax, ymax, factor, bins in zip(
            (0.4, 0.72, 0.08),
            ("speed_radius", "speed_average", "amplitude"),
            ("Speed radius (km)", "Speed average (cm/s)", "Amplitude (cm)"),
            (100, 50, 20),
            (0.08, 0.07, 0.4),
            (0.001, 100, 100),
            (np.arange(0, 2000, 1), np.arange(0, 1000, 0.5), np.arange(0.0005, 1000, 0.2)),
    ):
        for isu, suite in enumerate(suites):
            if isu == 0:
                ax_hist = fig.add_axes((x0, 0.24, 0.27, 0.35))

            if suite == 'aviso':
                col_a = 'red'
                col_c = 'blue'
            else:
                col_a = colours['anticyclonic'][isu]
                col_c = colours['cyclonic'][isu]

            print('bin edges ',len(bin_edges), len(bins), len(eddy_hist[(suite, name, 'anticyclonic')]))
            print('ax_hist ', x0, name, suite)
            ax_hist.plot(bins[1:], eddy_hist[(suite, name, 'anticyclonic')], color=col_a, label = suite+', AC')
            ax_hist.plot(bins[1:], eddy_hist[(suite, name, 'cyclonic')], color=col_c, label = suite+', C')
            ax_hist.set_xticklabels([])
            ax_hist.set_xlim(0, xmax)
            ax_hist.set_ylim(0, ymax)
            ax_hist.grid()

        for isu, suite in enumerate(suites):
            if suite == 'aviso':
                col_a = 'red'
                col_c = 'blue'
            else:
                col_a = colours['anticyclonic'][isu]
                col_c = colours['cyclonic'][isu]

            if isu == 0:
                ax_cum = fig.add_axes((x0, 0.62, 0.27, 0.35))
            ax_cum.plot(bins[1:], eddy_cum[(suite, name, 'anticyclonic')], color=col_a, label = suite+', AC')
            ax_cum.plot(bins[1:], eddy_cum[(suite, name, 'cyclonic')], color=col_c, label = suite+', C')
            ax_cum.set_xticklabels([])
            ax_cum.set_title(title+', '+str(nyears)+' years')
            ax_cum.set_xlim(0, xmax)
            ax_cum.set_ylim(0, 1)
            ax_cum.grid()

        for isu, suite in enumerate(suites):
            if isu == 0:
                ax_ratio = fig.add_axes((x0, 0.06, 0.27, 0.15))
            ax_ratio.set_xlim(0, xmax)
            ax_ratio.set_ylim(0, 2)
            if suite == 'aviso':
                col = 'blue'
            else:
                col = 'cyan'
            #ax_ratio.plot((bins[1:] + bins[:-1]) / 2, nb_c / nb_a, c=col)
            ax_ratio.plot((bins[1:] + bins[:-1]) / 2, eddy_hist[(suite, name, 'cyclonic')] / eddy_hist[(suite, name, 'anticyclonic')], c=col)
            ax_ratio.axhline(1, color='k')
            ax_ratio.grid()
            ax_ratio.set_xlabel(title)

    ax_cum.set_ylabel("Cumulative\npercent distribution")
    ax_hist.set_ylabel("Percent of observations")
    ax_ratio.set_ylabel("Ratio percent\nCyc/Acyc")
    ax_hist.legend()
    ax_cum.legend()

    suite_names = '_'.join(suites)
    #plt.suptitle('Eddy properties for '+suite_names+' over '+str(nyears)+' years')
    figname = os.path.join(fig_dir, 'eddy_properties_'+suite_names+'_new.png')
    plt.savefig(figname)
    plt.show()

def plot_eddy_properties_2(suites, min_lifetime, nyears):
    eddy_hist = {}; eddy_cum = {}
    for isu, suite in enumerate(suites):
        for name in eddy_prop_1:
            for eddy_t in eddy_types:
                pickle_file = os.path.join(eddy_prop_dir, suite+'_'+eddy_t+'_'+name+'_'+str(nyears)+'years_'+str(min_lifetime)+'minlifetime_hist.pkl')
                with open(pickle_file, 'rb') as fh:
                    hist = pickle.load(fh)
                    bin_edges = pickle.load(fh)
                    eddy_hist[(suite, name, eddy_t)] = hist
                dist = bin_edges[1] - bin_edges[0]
                cumsum = np.cumsum(eddy_hist[(suite, name, eddy_t)]*dist)
                eddy_cum[(suite, name, eddy_t)] = np.subtract(1.0, cumsum)

    fig = plt.figure(figsize=(14, 8))
    kwargs_a = dict(histtype="step", density=True)
    kwargs_c = dict(histtype="step", density=True)
    #hist_max = [0.08, 0.03, 0.4]

    for x0, name, title, xmax, ymax, factor, bins in zip(
            (0.05, 0.28, 0.51, 0.74),
            ("amplitude", "speed_radius", "speed_average", "effective_contour_shape_error" ),
            ("Amplitude (cm)", "Radius (speed) (km)", "Speed average (cm/s)", "Shape error" ),
            (20, 100, 50, 100),
            (0.4, 0.08, 0.07, 0.08),
            (100, 0.001, 100, 1),
            (np.arange(0.0005, 1000, 0.2), np.arange(0, 2000, 1), np.arange(0, 1000, 0.5), np.arange(0, 100, 1)),
    ):
        for isu, suite in enumerate(suites):
            if isu == 0:
                ax_hist = fig.add_axes((x0, 0.24, 0.19, 0.35))

            if suite == 'aviso':
                col_a = 'red'
                col_c = 'blue'
            else:
                col_a = colours['anticyclonic'][isu]
                col_c = colours['cyclonic'][isu]

            ax_hist.plot(bins[1:], eddy_hist[(suite, name, 'anticyclonic')], color=col_a, label = suite+', AC')
            ax_hist.plot(bins[1:], eddy_hist[(suite, name, 'cyclonic')], color=col_c, label = suite+', C')
            ax_hist.set_xticklabels([])
            ax_hist.set_xlim(0, xmax)
            ax_hist.set_ylim(0, ymax)
            ax_hist.grid(color='lightgray')
            if 'shape' in name:
                ax_hist.plot([70,70], [0,1], color='black')

        for isu, suite in enumerate(suites):
            if suite == 'aviso':
                col_a = 'red'
                col_c = 'blue'
            else:
                col_a = colours['anticyclonic'][isu]
                col_c = colours['cyclonic'][isu]

            if isu == 0:
                ax_cum = fig.add_axes((x0, 0.62, 0.19, 0.35))
            ax_cum.plot(bins[1:], eddy_cum[(suite, name, 'anticyclonic')], color=col_a, label = suite+', AC')
            ax_cum.plot(bins[1:], eddy_cum[(suite, name, 'cyclonic')], color=col_c, label = suite+', C')
            ax_cum.set_xticklabels([])
            ax_cum.set_title(title+', '+str(nyears)+' years')
            ax_cum.set_xlim(0, xmax)
            ax_cum.set_ylim(0, 1)
            ax_cum.grid(color='lightgray')
            if 'shape' in name:
                ax_cum.plot([70,70], [0,1], color='black')

        for isu, suite in enumerate(suites):
            if isu == 0:
                ax_ratio = fig.add_axes((x0, 0.06, 0.19, 0.15))
            ax_ratio.set_xlim(0, xmax)
            ax_ratio.set_ylim(0, 2)
            if suite == 'aviso':
                col = 'blue'
            else:
                col = 'cyan'
            ax_ratio.plot((bins[1:] + bins[:-1]) / 2, eddy_hist[(suite, name, 'cyclonic')] / eddy_hist[(suite, name, 'anticyclonic')], c=col)
            ax_ratio.axhline(1, color='k')
            ax_ratio.grid(color='lightgray')
            ax_ratio.set_xlabel(title)

        if name == 'amplitude':
            ax_cum.set_ylabel("Cumulative\npercent distribution")
            ax_hist.set_ylabel("Percent of observations")
            ax_ratio.set_ylabel("Ratio percent\nCyc/Acyc")
            ax_hist.legend()
            ax_cum.legend()

    suite_names = '_'.join(suites)
    #plt.suptitle('Eddy properties for '+suite_names+' over '+str(nyears)+' years')
    figname = os.path.join(fig_dir, 'eddy_properties_'+suite_names+'_'+str(nyears)+'years_'+str(min_lifetime)+'minlifetime.png')
    plt.savefig(figname)
    plt.show()

def work(suites):

    min_lifetime = min_lifetime_plot_tracks
    plot_tracks(suites, nyears, min_lifetime=min_lifetime)

    # calculates eddy birth/tracks on 2x2 degree grid (writes pkl files as intermediate)
    min_lifetime = min_lifetime_pdfs
    calc_birth(suites, min_lifetime, plot_type='density', nyears=nyears)
    calc_birth(suites, min_lifetime, plot_type='genesis', nyears=nyears)
    plot_birth(suites, min_lifetime, plot_type='density', nyears=nyears)
    plot_birth(suites, min_lifetime, plot_type='genesis', nyears=nyears)

    min_lifetime = min_lifetime_pdfs
    lifetime_histogram(suites, nyears, min_lifetime)

    # calculate eddy properties (which writes pkl files to be used by plot_eddy_properties
    min_lifetime = min_lifetime_pdfs
    for suite in suites:
        calculate_eddy_properties(suite, nyears, min_lifetime)
    plot_eddy_properties_2(suites, min_lifetime, nyears)

## old code or code yet to work
    #for suite in suites:
    #    plot_eddy_properties(eddy_subset, [suite])
    # get subset of tracks longer than min_length for plotting
    #eddy_subset = read_datafiles(suites, min_lifetime = min_lifetime_tracks)
    #plot_amplitude(suites, eddy_subset)
    #plot_radius(suites, eddy_subset)


if __name__ == '__main__':
    work(suites)

