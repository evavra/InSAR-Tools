#!/bin/csh -f
# 
#  Modified by Ellis Vavra, 12/3/2020
#  
#  Preprocess all the data based on data.in table file and generate: 
#  1. SLC files
#  2. PRM files 
#  3. LED files
#  4. time-baseline plot for user to create stacking pairs 

#  format in data.in table file: 
#  	line 1: master_name  
# 	line 2 and below: aligned_name
alias rm 'rm -f'
unset noclobber

#
# check the number of arguments 
# 
if ($#argv != 3) then 
echo ""
echo "Usage: pre_proc_batch_ALOS2_SCAN.csh data.in batch.config [swaths]"
echo "       preprocess a set of images using a common radius"
echo ""
echo "       Format of data.in is:"
echo "         line 1: master_name "
echo "         line 2 and below: aligned_name"
echo ""
echo "       Example of data.in for ALOS2 SCANSAR is:"
echo "         IMG-HH-ALOS2047473650-150410-WBDR1.1__D"
echo "         IMG-HH-ALOS2101293650-160408-WBDR1.1__D"
echo ""
echo "       Optional: specify swaths for processing separated by commas 1,2,3,4,5" 
echo ""     
echo "Example: pre_proc_batch_ALOS2_SCAN.csh data.in batch.config 2,3"
echo ""
exit 1
endif

# read parameters from configuration file
set earth_radius = `grep earth_radius $2 | awk '{print $3}'`
set SLC_factor = `grep SLC_factor $2 | awk '{print $3}'`

set commandline = ""
if (!($earth_radius == "")) then
    set commandline = "$commandline -radius $earth_radius"
endif

if (!($SLC_factor == "")) then
    set commandline = "$commandline -SLC_factor $SLC_factor"
endif

echo $commandline

# open and read data.in table 
echo ""
echo "START PREPROCESS A STACK OF IMAGES"
echo ""
echo "Preprocess master image"

set line1 = `awk 'NR==1 {print $0}' $1`
set master = `echo $line1[1] | awk '{ print substr($1,8,length($1)-7)}'`


#
# loop over specified subswaths
#
set swath_list = `echo $3:q | sed 's/,/ /g'`

if (${#swath_list} < 1) then
    set swath_list = (1 2 3 4 5)
endif

echo " Using swaths $swath_list"

foreach subswath ($swath_list)
    
    # unpack the master if necessary
    if(! -f IMG-HH-$master-F$subswath.SLC || ! -f IMG-HH-$master-F$subswath.PRM ) then
        echo "ALOS_pre_process_SLC IMG-HH-$master-F$subswath LED-$master -ALOS2 $commandline"
        ALOS_pre_process_SLC IMG-HH-$master-F$subswath LED-$master -ALOS2 $commandline
    endif

    set RAD = `grep earth_radius IMG-HH-$master-F$subswath.PRM | awk '{print $3}'`
    set rng_samp_rate_m = `grep rng_samp_rate IMG-HH-$master-F$subswath.PRM | awk 'NR == 1 {printf("%d", $3)}'`

    baseline_table.csh IMG-HH-$master-F$subswath.PRM IMG-HH-$master-F$subswath.PRM >! baseline_table.dat
    baseline_table.csh IMG-HH-$master-F$subswath.PRM IMG-HH-$master-F$subswath.PRM GMT >! table.gmt
    
    # loop and unpack the aligned image using the same earth radius as the master image
    foreach line2 (`awk 'NR>1 {print $0}' $1`)
        
        echo "pre_proc_batch.csh"
        echo "preprocess aligned images"
         
        set aligned = ` echo $line2 | awk '{ print substr($1,8,length($1)-7)}'`
        if(! -f IMG-HH-$aligned-F$subswath.SLC || ! -f IMG-HH-$aligned-F$subswath.PRM ) then
            echo "ALOS_pre_process_SLC IMG-HH-$aligned-F$subswath LED-$aligned -ALOS2 -radius $RAD -SLC_factor $SLC_factor"
            ALOS_pre_process_SLC IMG-HH-$aligned-F$subswath LED-$aligned -ALOS2 -radius $RAD -SLC_factor $SLC_factor
        endif
        
        # check the range sampling rate of the aligned images and do conversion if necessary
        set rng_samp_rate_s = `grep rng_samp_rate IMG-HH-$aligned-F$subswath.PRM | awk 'NR == 1 {printf("%d", $3)}'`
        set t = `echo $rng_samp_rate_m $rng_samp_rate_s | awk '{printf("%1.1f\n", $1/$2)}'`
        
        if ($t == 1.0) then
            echo "The range sampling rate for master and aligned images are: "$rng_samp_rate_m

            baseline_table.csh IMG-HH-$master-F$subswath.PRM IMG-HH-$aligned-F$subswath.PRM >> baseline_table.dat
            baseline_table.csh IMG-HH-$master-F$subswath.PRM IMG-HH-$aligned-F$subswath.PRM GMT >> table.gmt

        else if ($t == 2.0) then
            echo "Convert the aligned image from FBD to FBS mode"
            
            ALOS_fbd2fbs_SLC IMG-HH-$aligned-F$subswath.PRM IMG-HH-$aligned-F$subswath"_"FBS.PRM
            baseline_table.csh IMG-HH-$master-F$subswath.PRM IMG-HH-$aligned-F$subswath"_"FBS.PRM >> baseline_table.dat
            baseline_table.csh IMG-HH-$master-F$subswath.PRM IMG-HH-$aligned-F$subswath"_"FBS.PRM GMT >> table.gmt
            
            echo "Overwriting the old aligned image"
            
            mv IMG-HH-$aligned-F$subswath"_"FBS.PRM IMG-HH-$aligned-F$subswath.PRM
            update_PRM IMG-HH-$aligned-F$subswath.PRM input_file IMG-HH-$aligned-F$subswath.SLC
            mv IMG-HH-$aligned-F$subswath"_"FBS.SLC IMG-HH-$aligned-F$subswath.SLC

        else if  ($t == 0.5) then
          echo "Use FBS mode image as master"
          exit 1

        else
          echo "The range sampling rate for master and aligned images are not convertable"
          exit 1
        endif


    end
end

# make baseline plots
awk '{print 2014.5+($1-181)/365.25,$2,$7}' < table.gmt > text
#  awk '{print 2014.5+($1-181)/365.25,$2,9,$4,$5,$6,$7}' < table.gmt > text
set region = `gmt gmtinfo text -C | awk '{print $1-0.5, $2+0.5, $3-500, $4+500}'`
gmt pstext text -JX8.8i/6.8i -R$region[1]/$region[2]/$region[3]/$region[4] -D0.2/0.2 -X1.5i -Y1i -K -N -F+f8,Helvetica+j5 > stacktable_all.ps
awk '{print $1,$2}' < text > text2
gmt psxy text2 -Sp0.2c -G0 -R -JX -Ba1:"year":/a200g00f100:"baseline (m)":WSen -O >> stacktable_all.ps

echo ""
echo "END PREPROCESS A STACK OF IMAGES"
echo ""

# clean up the mess
rm text text2 
