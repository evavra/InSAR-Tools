#!/bin/bash
#
# clip_intf.sh
# July 8, 2019 - Ellis Vavra
#
  if ($#argv < 6) then
errormessage:
    echo ""
    echo "inft_clip.csh [GMTSAR] - Unwrap the phase with nearest neighbor interpolating low coherence and blank pixels"
    echo ""
    echo "Usage: inft_clip.csh input_filename output_filename minium_range maximum_range minimum_azimuth maximum_azimuth "
    echo ""
    echo "       Run from parent directory of GMTSAR interferogram directories, usually intf_all (i.e. intf_all/2019001_20190025/datafiles)"
    echo ""
    echo "Example: snaphu.csh phasefilt.grd 0 20000 0 5000"
    exit 1
  endif
#
# 1. Set input options from argument list
set input=$1    # name of input grid file
set output=$2   # name of output grid file
set minRa=$3    # min. range for clipping
set maxRa=$4    # max. azimuth for clipping
set minAz=$5    # min. range for clipping
set maxAz=$6    # max. azimuth for clipping



# 2. Loop through each sub-directory

for directory in `ls -1 -d *`
do
    echo $directory

done

# For each directory clip files that match specified name string to the dimensions given in the arguments

