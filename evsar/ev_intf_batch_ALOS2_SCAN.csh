#!/bin/csh -f 
#       $Id$
# 
# Loop through a list of interferometry pairs
# modified from process2pass.csh
# 
# Modified by Ellis Vavra, 12/12/2020

alias rm 'rm -f'
unset noclobber

# 
if ($#argv < 2) then 
    echo ""
    echo "Usage: intf_batch_ALOS2_SCAN.csh intf.in batch.config [swaths]"
    echo "  make a stack of interferograms listed in intf.in"
    echo ""
    echo " SAT can only be ALOS " 
    echo ""
    echo " format for intf.in:"
    echo "     reference1_name:repeat1_name"
    echo "     reference2_name:repeat2_name"
    echo "     reference3_name:repeat3_name"
    echo "     ......"
    echo ""
    echo " Example of intf.in for ALOS2 ScanSAR"
    echo "    IMG-HH-ALOS2101532950-160409-WBDR1.1__D:IMG-HH-ALOS2149142950-170225-WBDR1.1__D"
    echo "    IMG-HH-ALOS2101532950-160409-WBDR1.1__D:IMG-HH-ALOS2155352950-170408-WBDR1.1__D"
    echo ""
    echo " Optional: specify swaths for processing separated by commas 1,2,3,4,5" 
    echo ""
    echo " batch.alos2.config is a config file for making interferograms"
    echo " See example.batch.config for an example"
    echo ""
    echo ""
    echo ""
    exit 1
endif

# Should fix this down below sometime
set SAT = 'ALOS'

# Make sure the files exist
if (! -f $1) then 
    echo "no input file:" $1
    exit
endif

if (! -f $2) then
    echo "no config file:" $2
    exit
endif

# Read parameters from configuration file
set stage = `grep proc_stage $2 | awk '{print $3}'`

if ($SAT == ALOS) then
    set master = `grep master_image $2 | awk '{print $3}' | awk '{ print substr($1,8,length($1)-7)}'`
endif

if ($master == "") then
    echo ""
    echo "master image not set."
    echo ""
    exit 1
endif

# If filter wavelength is not set then use a default of 200m
set filter = `grep filter_wavelength $2 | awk '{print $3}'`
if ( "x$filter" == "x" ) then
    set filter = 200
    echo " "
    echo "WARNING filter wavelength was not set in config.txt file"
    echo "        please specify wavelength (e.g., filter_wavelength = 200)"
    echo "        remove filter1 = gauss_alos_200m"
endif

# Asign other parameters
set dec = `grep dec_factor $2 | awk '{print $3}'` 
set topo_phase = `grep topo_phase $2 | awk '{print $3}'` 
set shift_topo = `grep shift_topo $2 | awk '{print $3}'` 
set threshold_snaphu = `grep threshold_snaphu $2 | awk '{print $3}'`
set threshold_geocode = `grep threshold_geocode $2 | awk '{print $3}'`
set switch_land = `grep switch_land $2 | awk '{print $3}'`
set defomax = `grep defomax $2 | awk '{print $3}'`
set range_dec = `grep range_dec $2 | awk '{print $3}'`
set azimuth_dec = `grep azimuth_dec $2 | awk '{print $3}'`
set near_interp = `grep near_interp $2 | awk '{print $3}'`
# set region_cut = `grep region_cut $2 | awk '{print $3}'`

##########################################
#            Begin processing            #
##########################################

# Pull swaths or set to all five
set swath_list = `echo $3:q | sed 's/,/ /g'`

if (${#swath_list} < 1) then
    set swath_list = (1 2 3 4 5)
endif

echo " Using swaths $swath_list"

# Loop over subswaths
foreach subswath ($swath_list)

    mkdir -p F$subswath    
    set region_cut = `grep region_cut $2 | awk '{print $3}'`
    cd F$subswath
    mkdir -p topo
    mkdir -p SLC

    # Link SLC files from raw directory
    cd SLC
    
    ln -s ../../raw/IMG-HH-*-F$subswath.LED .
    ln -s ../../raw/IMG-HH-*-F$subswath.PRM .
    ln -s ../../raw/IMG-HH-*-F$subswath.SLC .
    
    cd ..

    ##################################
    # 1 - start from make topo_ra    #
    ##################################

    if ($stage <= 1) then
        
        # Clean up
        cleanup.csh topo
        
        # make topo_ra
        if ($topo_phase == 1) then

            echo " "
            echo "DEM2TOPOPHASE.CSH - START"
            echo "USER SHOULD PROVIDE DEM FILE"
            cd topo

            if ($SAT == ALOS) then
                # echo `pwd`
                # echo ../SLC/IMG-HH-$master-F$subswath.PRM

                cp ../SLC/IMG-HH-$master-F$subswath.PRM master.PRM
                ln -s ../SLC/IMG-HH-$master-F$subswath.LED .
            endif

            if (-f ../../topo/dem.grd) then
                ln -s ../../topo/dem.grd . 
                dem2topo_ra_ALOS2.csh master.PRM dem.grd
            else
                echo "no DEM file found: " dem.grd
                exit 1
            endif

            cd ..

            echo "DEM2TOPOPHASE.CSH - END"
            #
            # shift topo_ra
            #
            if ($shift_topo == 1) then
                echo " "
                echo "OFFSET_TOPO - START"
                cd SLC
                if ($SAT == ALOS) then
                    set rng_samp_rate = `grep rng_samp_rate IMG-HH-$master-F$subswath.PRM | awk 'NR == 1 {printf("%d", $3)}'`
                    if ( $?rng_samp_rate) then
                        if ($rng_samp_rate > 25000000) then
                            echo "processing ALOS FBS data"
                            set rng = 2
                        else
                            echo "processing ALOS FBD data"
                            set rng = 1
                        endif
                    else
                        echo "Undefined rng_samp_rate in the master PRM file"
                        exit 1
                    endif

                    slc2amp.csh IMG-HH-$master-F$subswath.PRM $rng amp-$master.grd

                else if ($SAT == ERS || $SAT == ENVI) then
                    slc2amp.csh $master.PRM 1 amp-$master.grd
                endif

                cd ..
                cd topo
                ln -s ../SLC/amp-$master.grd .
                offset_topo amp-$master.grd topo_ra.grd 0 0 7 topo_shift.grd
                cd ..
                echo "OFFSET_TOPO - END"

            else if ($shift_topo == 0) then
                echo "NO TOPOPHASE SHIFT "
            else
                echo "Wrong paramter: shift_topo "$shift_topo
                exit 1
            endif

        else if ($topo_phase == 0) then
            echo "NO TOPOPHASE IS SUBSTRACTED"
        else
            echo "Wrong paramter: topo_phase "$topo_phase
            exit 1
        endif # Not sure
        endif # what's going on here
    endif 

    ##################################################
    # 2 - start from make and filter interferograms  #
    #                unwrap phase and geocode        #
    ##################################################

    if ($stage <= 2) then
    #
    # make working directories
    #
    echo ""
    echo "START FORM A STACK OF INTERFEROGRAMS"

    mkdir -p intf/
    ln -s ../$1 . 
    
    # loop over intf.in
    #
    foreach line (`awk '{print $0}' $1`)

        if ($SAT == ALOS) then
            set ref = `echo $line | awk -F: '{print $1}' | awk '{ print substr($1,8,length($1)-7)}'`
            set rep = `echo $line | awk -F: '{print $2}' | awk '{ print substr($1,8,length($1)-7)}'`

            set ref_id  = `grep SC_clock_start ./SLC/IMG-HH-$ref-F$subswath.PRM | awk '{printf("%d",int($3))}' `
            set rep_id  = `grep SC_clock_start ./SLC/IMG-HH-$rep-F$subswath.PRM | awk '{printf("%d",int($3))}' `

            echo ""
            echo "INTF.CSH, FILTER.CSH - START"
            cd intf
            rm -r $ref_id"_"$rep_id
            mkdir $ref_id"_"$rep_id
            cd $ref_id"_"$rep_id
            ln -s ../../SLC/IMG-HH-$ref-F$subswath.LED .
            ln -s ../../SLC/IMG-HH-$rep-F$subswath.LED .
            ln -s ../../SLC/IMG-HH-$ref-F$subswath.SLC .
            ln -s ../../SLC/IMG-HH-$rep-F$subswath.SLC .
            cp ../../SLC/IMG-HH-$ref-F$subswath.PRM .
            cp ../../SLC/IMG-HH-$rep-F$subswath.PRM .
        endif

        if ($SAT == ALOS) then 
            echo "Forming interogram IMG-HH-$ref-F$subswath.PRM IMG-HH-$rep-F$subswath.PRM..."

            if($topo_phase == 1) then
                if ($shift_topo == 1) then
                    echo "Shifting topo_ra..."
                    ln -s ../../topo/topo_shift.grd .
                    intf.csh IMG-HH-$ref-F$subswath.PRM IMG-HH-$rep-F$subswath.PRM -topo topo_shift.grd
                         # filter.csh IMG-HH-$ref-F$subswath.PRM IMG-HH-$rep-F$subswath.PRM $filter $dec 2 8
                    filter.csh IMG-HH-$ref-F$subswath.PRM IMG-HH-$rep-F$subswath.PRM $filter $dec $range_dec $azimuth_dec
                else
                    echo "Subtracting topographic phase..."
                    ln -s ../../topo/topo_ra.grd .
                    intf.csh IMG-HH-$ref-F$subswath.PRM IMG-HH-$rep-F$subswath.PRM -topo topo_ra.grd
                         # filter.csh IMG-HH-$ref-F$subswath.PRM IMG-HH-$rep-F$subswath.PRM $filter $dec 2 8
                    filter.csh IMG-HH-$ref-F$subswath.PRM IMG-HH-$rep-F$subswath.PRM $filter $dec $range_dec $azimuth_dec
                endif
            else
                intf.csh IMG-HH-$ref-F$subswath.PRM IMG-HH-$rep-F$subswath.PRM
                       # filter.csh IMG-HH-$ref-F$subswath.PRM IMG-HH-$rep-F$subswath.PRM $filter $dec 2 8
                filter.csh IMG-HH-$ref-F$subswath.PRM IMG-HH-$rep-F$subswath.PRM $filter $dec $range_dec $azimuth_dec
            endif
        endif

        echo "INTF.CSH, FILTER.CSH - END"


        # If region cut isn't specified, get bounds of frame
        if ($region_cut == "") then
            set region_cut = `gmt grdinfo phase.grd -I- | cut -c3-20`
            set rs = `gmt grdinfo phase.grd -C | awk '{print $2}'`
            set re = `gmt grdinfo phase.grd -C | awk '{print $3-16}'`
            set as = `gmt grdinfo phase.grd -C | awk '{print $4}'`
            set ae = `gmt grdinfo phase.grd -C | awk '{print $5}'`
            set region_cut = "$rs/$re/$as/$ae"
        endif


        if ($threshold_snaphu != 0 ) then
            # Generate coherence mask if specified
            if ($switch_land == 1) then
                cd ../../topo

                if (! -f landmask_ra.grd) then
                    landmask_ALOS2.csh $region_cut
                endif

                cd ../intf
                cd $ref_id"_"$rep_id
                ln -s ../../topo/landmask_ra.grd .
            endif

            # Do unwrapping
            echo ""
            echo "SNAPHU.CSH - START"
            echo "threshold_snaphu: $threshold_snaphu"
            if ($near_interp == 1) then
                snaphu_interp.csh $threshold_snaphu $defomax $region_cut
            else
                snaphu.csh $threshold_snaphu $defomax $region_cut
            endif
            echo "SNAPHU.CSH - END"

        else 
            echo ""
            echo "SKIP UNWRAP PHASE"
        endif

        echo " "
        echo "GEOCODE.CSH - START"
        rm raln.grd ralt.grd

        if ($topo_phase == 1) then
            rm trans.dat
            ln -s  ../../topo/trans.dat . 
            echo "threshold_geocode: $threshold_geocode"
            geocode.csh $threshold_geocode
        else 
            echo "topo_ra is needed to geocode"
            exit 1
        endif
        echo "GEOCODE.CSH - END"

        cd ../..

    end # loop of foreach 
    endif # stage 2

    # end of subswath loop
    cd ..
end

echo ""
echo "END FORM A STACK OF INTERFEROGRAMS"
echo ""

