#!/bin/tcsh -f
# Ellis Vavra, Jan. 2021

if ($#argv < 3) then
    echo ""
    echo "Rename set of grd files."
    echo ""
    echo "Usage: batch_rename.csh intf_list curent_name new_name"
    echo ""
    echo "intf_list  - list of interferogram directories"
    echo "    e.g."
    echo "    date1_date2"
    echo "    date2_date3"
    echo "    date3_date4"
    echo "    ......"
    echo "intf_dir   - path to directory containing interferograms"
    echo "current_name  - current filestem"
    echo "new_name   -  new filestem"
    echo ""
    exit 1
endif


set intf_list = $1
set intf_dir  = $2
set file_type = $3
set new_file  = $4
set region    = $5


# Loop over files. Run everything from top-level directory
foreach intf (`cat $intf_list`)
    mv $intf_dir/$intf/$current_name.grd $intf_dir/$intf/$new_name.grd 
end