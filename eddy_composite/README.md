# eddy_composites

Scripts and notebooks for building eddy composites with example over Agulhas rings
©2023 MPI-M, Dian Putrasahan

1) Remap data onto MR25 grid
	cd /home/m/m300466/NGC2/eddytrack
	>> sbatch remap_mm_r2b8O_MR25.job zos ngc2013

2) Split files into yyyymm 
	Edit  split_mm_yrmth.sh
	varname=zos
	varname=Wind_Speed_10m
	varname=atmos_fluxes_HeatFlux_Latent
	varname=mlotst
	varname=sea_level_pressure
	varname=to
	expid=ngc2013
	expid=rthk001
	>> ./split_mm_yrmth.sh

3) Eddy identification to get anticyclones and cyclones (only with zos)
	cd /home/m/m300466/NGC2/eddytrack
	Notebook with some figures to show identified eddies [eddy-identification.ipynb]
	#Submit scripts for each year using submit_IDeddy_mm.job, which in turn submits IDeddy_mm.py script for each month
	>> for yr in $(seq 2020 2049); do sbatch submit_IDeddy_mm.job ${yr}; done

4) Remap and high pass filter other fields
	cd /home/m/m300466/NGC2/eddytrack
	>> sbatch remap_mm_r2b8O_MR25.job Wind_Speed_10m ngc2013
	>> sbatch remap_mm_r2b8O_MR25.job atmos_fluxes_HeatFlux_Latent ngc2013
	>> sbatch remap_mm_r2b8O_MR25.job mlotst ngc2013
	>> sbatch remap_mm_r2b8O_MR25.job sea_level_pressure ngc2013
	>> sbatch remap_mm_r2b8O_MR25.job to ngc2013
	# Submit scripts for each year using submit_highpass_mm.job, which in turn submits highpass_mm.py script
	>> for yr in $(seq 2020 2049); do sbatch submit_highpass_mm.job zos ${yr}; done
	>> for yr in $(seq 2020 2050); do sbatch submit_highpass_mm.job Wind_Speed_10m ${yr}; done
	>> for yr in $(seq 2020 2050); do sbatch submit_highpass_mm.job atmos_fluxes_HeatFlux_Latent ${yr}; done
	>> for yr in $(seq 2020 2050); do sbatch submit_highpass_mm.job to ${yr}; done
	>> for yr in $(seq 2020 2050); do sbatch submit_highpass_mm.job atmos_fluxes_FrshFlux_Precipitation ${yr}; done
	>> for yr in $(seq 2020 2050); do sbatch submit_highpass_mm.job mlotst ${yr}; done

5) Composite for a given region (Agulhas rings)
monthly-eddy-composites.ipynb


