#!/bin/bash


dataDir=/Users/ellisvavra/Thesis/insar/des/f2/intf_all/Attempt15-Cmin-0.20/SBAS_SMOOTH_0.0000e+00/
dataDirLen=`echo -n $dataDir | wc -c`

format=LOS*INT3.grd
lookupTable=/Users/ellisvavra/Thesis/insar/des/f2/intf_all/Geocoding/trans.dat
colorMap=/Users/ellisvavra/Thesis/insar/des/f2/intf_all/Geocoding/LOS.cpt

outDir=/Users/ellisvavra/Thesis/insar/des/f2/intf_all/Geocoding/Attempt15/

# Project each grid to lat/long coordinate system
for filePath in `ls $dataDir$format`
do 
    echo 'Projecting' $filePath 'to ll...'
    proj_ra2ll.csh $lookupTable $filePath $outDir${filePath: $dataDirLen : 17}_ll.grd
    # echo 'proj_ra2ll.csh' $filePath $outDir${filePath: $dataDirLen : 18}_ll.grd 

done

