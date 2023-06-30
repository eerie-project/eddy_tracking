#!/bin/bash
# set -ex

#------------------------
# users choice
#------------------------

#varname=zos
#varname=Wind_Speed_10m
#varname=atmos_fluxes_HeatFlux_Latent
varname=mlotst
#varname=to
#varname=atmos_fluxes_FrshFlux_Precipitation

expid=ngc2013
#expid=rthk001

freqm=mm
zidx=1
outdir=/work/mh0287/m300466/topaz/${expid}/${varname}/${freqm}

#========================
YYYYs=2020
MMs=2
YYYYe=2050
MMe=1
YYYYint=$YYYYs
declare -a datearr
declare -a dayarr
declare -a yyarr
declare -a mmarr
#declare -a ddarr
#dayarr=()

while [ $YYYYint -le $YYYYe ]; do
  if [ $YYYYint -eq $YYYYs ]; then
     MMint=$MMs
  else
     MMint=1
  fi
  if [ $YYYYint -eq $YYYYe ]; then
     MMend=$MMe
  else
     MMend=12
  fi
  while [ $MMint -le $MMend ]; do
     if [ $MMint -le 9 ]; then
        MMint=`eval echo "0$MMint"`
     fi
     #echo "Extract for yr$YYYYint mth$MMint day$DDint"
     #dayarr+=( $fdate )
     datearr+=( "${YYYYint}-${MMint}" )
     dayarr+=( "${YYYYint}${MMint}" )
     yyarr+=( "${YYYYint}" )
     mmarr+=( "${MMint}" )
     #ddarr+=( "${DDint}" )
     MMint=`expr $MMint + 01`
  done
  YYYYint=`expr $YYYYint + 01`
done

#array length
echo ${#dayarr[@]}
##array list
#echo ${dayarr[@]}

for tidx in $(seq 0 ${#dayarr[@]});
do
  echo "ddate="${datearr[$tidx]}
  ddate=${datearr[$tidx]}
  echo "fdate="${dayarr[$tidx]}
  fdate=${dayarr[$tidx]}
  yyyy=${yyarr[$tidx]}
  mm=${mmarr[$tidx]}
  #Remove depth dimension in SST
  varfile1=${outdir}/${varname}_${freqm}_${fdate}.nc
  if [ ${varname} == 'to' ]; then
     bigfile=${outdir}/${expid}_${varname}_${zidx}_${freqm}_202002-205001_MR25.nc
     file2sm=${outdir}/${expid}_${varname}_${zidx}_${freqm}_${fdate}_MR25.nc
     cdo -L -select,year=${yyyy},month=${mm} ${bigfile} ${varfile1}
     ncwa -a depth ${varfile1} ${file2sm}
     ncks -C -O -x -v depth ${file2sm} ${file2sm}
     rm $varfile1
  else
     bigfile=${outdir}/${expid}_${varname}_${freqm}_202002-205001_MR25.nc
     file2sm=${outdir}/${expid}_${varname}_${freqm}_${fdate}_MR25.nc
     cdo -L -select,year=${yyyy},month=${mm} ${bigfile} ${file2sm}
     #ncks -C -O -x -v lon,lat ${file2sm} ${file2sm}
  fi
done


