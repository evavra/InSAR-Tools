#!/bin/tcsh -f
# Ellis Vavra, Jan. 2021

if ($#argv < 3) then
    echo ""
    echo "Cut set of grd files using grdcut."
    echo ""
    echo "Usage: batch_cut.csh intf_list file_type new_file region"
    echo ""
    echo "intf_list  - list of interferogram directories"
    echo "    e.g."
    echo "    date1_date2"
    echo "    date2_date3"
    echo "    date3_date4"
    echo "    ......"
    echo "intf_dir   - path to directory containing interferograms"
    echo "file_type  - filestem of product to cut (e.g. phase, phasefilt, corr)"
    echo "new_file   - filestem for cut grids"
    echo "region     - x_min/x_max/y_min/y_max (e.g. 0/10000/20000/40000"
    echo ""
    exit 1
endif


set intf_list = $1
set intf_dir  = $2
set file_type = $3
set new_file  = $4
set region    = $5


# Loop over files. Run everything from top-level directory
foreach intf (cat $intf_list)
    echo "Cutting $intf..."
    echo "grdcut $intf_dir/$file_type.grd -G$intf_dir/$new_file.grd -R$region"
end