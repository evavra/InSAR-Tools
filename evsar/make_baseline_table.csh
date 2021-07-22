#!/bin/tcsh -f
# Modified by Zeyu Jin on Jan. 2019
# Modified by Ellis Vavra on July 2021
# need to link all the original IMG and LED files
# plus the LED.list

alias rm 'rm -f'
unset noclobber

if ($#argv < 2) then
  echo "Makes baseline_table and baseline plot from PRM and LED files only (IMG files not needed)."
  echo ""
  echo "Usage: make_baseline_table.csh LED.list config_file [n1] [n2]"
  echo "  Run from top-level processing directory."
  echo ""
  echo "Inputs:"
  echo "  LED.list    - list of LED files for each scene to process"
  echo "  config_file - batch configuration file (example in repository)"
  echo "  (n1, n2)    - subswaths to be processed (n2 >= n1). Default is 1-5."
  echo ""

  exit 1
endif

# Specify subswaths
if ($#argv == 4) then
  set n1 = $3
  set n2 = $4
else 
  set n1 = 1
  set n2 = 5
endif

# make sure the files exist
if (! -f $2) then
   echo "No configure file: "$2
   exit
endif

foreach ledfile (`cat raw/$1`)
    if (! -f raw/$ledfile) then
      echo "No file raw/"$ledfile
      exit
    endif

    # set stem = `echo $ledfile|cut -c 5-`

    # set nf = $n1
    # while ($nf <= $n2)
    #     set imgfile = "IMG-HH-"$stem"-F"$nf
    #     if (! -f raw/$imgfile) then
    #      echo "No file raw/"$imgfile
    #      exit
    #     endif
    #     @ nf = $nf + 1
    # end
end

echo "Pre-checking files finished!"
# unset ledfile imgfile
unset ledfile

# read parameters from configure file
# if ($#argv == 4) then
#   set n1 = $3
#   set n2 = $4
# else 
#   set n1 = 1
#   set n2 = 5
# endif

set config = $2

set num_patches = `grep num_patches $config | awk '{print $3}'`
set SLC_factor = `grep SLC_factor $config | awk '{print $3}'`
set earth_radius = `grep earth_radius $config | awk '{print $3}'`
set topo_phase = `grep topo_phase $config | awk '{print $3}'`
set shift_topo = `grep shift_topo $config | awk '{print $3}'`

    set commandline = ""

    if (!($earth_radius == "")) then
      set commandline = "$commandline -radius $earth_radius"
    endif
    if (!($num_patches == "")) then
      set commandline = "$commandline -npatch $num_patches"
    endif
    if (!($SLC_factor == "")) then
      set commandline = "$commandline -SLC_factor $SLC_factor"
    endif

# # Actual SLC precprocessing
# echo "options for ALOS_preprocess_SLC: " $commandline

# set nf = $n1
# while ($nf <= $n2)
  
#   # make working directories
#   echo ".....processing F$nf"
# #  mkdir -p "F"$nf
# #  cd "F"$nf
# #  mkdir intf/ SLC/ topo/
# #  cd ..

#   # clean up last residuals
# #  rm raw/*"-F$nf.PRM"*
# #  rm raw/*"-F$nf.SLC"
# #  rm raw/*"-F$nf.LED"
  
#   # preprocess the raw data
#   cd raw
#   foreach ledfile (`cat $1`)
#      set stem = `echo $ledfile|cut -c 5-`
#      set imgfile = "IMG-HH-"$stem"-F"$nf
#      set ledfile = $ledfile
#      echo $imgfile
#      ALOS_pre_process_SLC $imgfile $ledfile $commandline
#   end
#   @ nf = $nf + 1
#   cd ..
# end

cd raw
ls -1 IMG*$n1*.PRM > prm.list
set master = `cat prm.list|awk '{ if (NR==1) print $1}'|awk -F".N1" '{print $1}'`
baseline_table.csh $master $master > baseline_table.dat


set yr_sar = `grep SC_clock_start $master | awk '{print $3}'|awk '{print substr($1,1,4)}'`
set day_sar = `grep SC_clock_start $master | awk '{print $3}'|awk '{print substr($1,5,3)}'`
day2date.csh $yr_sar$day_sar > date_sar.txt
set tsar = `echo $yr_sar $day_sar|awk '{printf "%.4f\n", $1+$2/365.25}'`
echo $tsar > tsar.tmp

foreach slave (`cat prm.list |awk '{if (NR>1) print $1}'|awk -F".N1" '{print $1}'`)
  baseline_table.csh $master $slave >> baseline_table.dat
  set yr_sar = `grep SC_clock_start $slave | awk '{print $3}'|awk '{print substr($1,1,4)}'`
  set day_sar = `grep SC_clock_start $slave | awk '{print $3}'|awk '{print substr($1,5,3)}'`
  day2date.csh $yr_sar$day_sar >> date_sar.txt
  set tsar = `echo $yr_sar $day_sar|awk '{printf "%.4f\n",$1+$2/365.25}'`
  echo $tsar >> tsar.tmp
end

cat prm.list | cut -c 5- | awk -F"-F" '{print $1}'  > img_id.list
paste date_sar.txt img_id.list baseline_table.dat |sort -u -k1,1 > baseline_table_new.dat

rm -f tsar.tmp
rm -f date_sar.txt
cd ..
