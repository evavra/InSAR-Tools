#!/bin/tcsh -f
# Ellis Vavra, Jan. 2021

if ($#argv < 3) then
  echo ""
  echo "Process unfiltered phase interferograms in parallel."
  echo "Requires real.grd and imag.grd in each interferogram directory specified in intf.list "
  echo "Run in directory above subswaths."
  echo ""
  echo "Usage: intf_batch_raw.csh intf.list swath Ncores"
  echo ""
  echo ""
  echo "Format for intf.list:"
  echo "  date1_date2"
  echo "  date2_date3"
  echo "  date3_date4"
  echo "  ......"
  echo ""
  echo ""
  exit 1
endif

rm -f phaseraw.cmd

# Start clock
set t1 = `date`

if ($#argv < 4) then
   set ncores = 4
else 
   set ncores = $3
endif

set swath = $2
# cd F$swath
# mkdir -p intf
# cleanup.csh intf
# cd ..

foreach intf (`cat $1`)
   # set date1 =  `echo $file1 |awk -F"-" '{print $4}'`
   # set date2 =  `echo $file2 |awk -F"-" '{print $4}'`
   set logfile = "phaseraw_"$intf"_F"$swath".log"
   # echo "intf_ALOS2_p2p_new.csh $file1 $file2 $2 $swath >& $logfile" >> intf_alos.cmd
   echo "gmt grdmath F$swath/$intf/imag.grd F$swath/$intf/real.grd ATAN2 FLIPUD = F$swath/$intf/phaseraw.grd >& $logfile" >> phaseraw.cmd
end

parallel --jobs $ncores < phaseraw.cmd
mv "phaseraw_"*"F$swath.log" F$swath

set t2 = `date`
set dir0 = `pwd`
echo "Job started on $t1 and finished on $t2 at $dir0/F$swath " | mail -s "Job finished - intf_batch_raw.csh" evavra@ucsd.edu
