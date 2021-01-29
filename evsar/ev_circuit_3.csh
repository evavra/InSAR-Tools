#!/bin/csh -f
#       $Id$
#
# Yuri Fialko, Nov 29, 2018
# Edited by Ellis Vavra, Jan. 28, 2021
# calculate phase circuit using triplets for raw and filtered phase given a list of interferograms
#

if ($#argv != 3) then
    echo ""
    echo "Usage: circuit_3.csh intf.list ftype dir landmask"
    echo "  Calculate phase circuit for raw and filtered phase "
    echo ""
    echo "  format of intf.list:"
    echo "    date1_date2  (list of sequential interferograms)"
    echo ""
    echo "  example of dates.short"
    echo "    20150626_20150601"
    echo ""
    echo "  ftype: file name (e.g., phasefilt or phaseraw_sub)"
    echo "  landmask: name of land mask file"
    echo ""
    echo "  outputs:"
    echo "    date1_date2_closure_raw.grd date1_date2_closure_filt.grd "
    echo ""
    exit 1
endif

# set up input parameters
set slist = $1 # dates.short
set ftype = $2 # filetype
set dir1  = $3 # interferogram directory
set dir2  = $3 # also interferogram directory
set mskf = "$dir2/topo/$landmask"
# set mskf = $dir2"/landmask_ra0.grd"

set ph0  = "ph.grd"

set out = "loop.out"
rm -f $out
set pi2=`echo "scale=7; 8*a(1)" | bc -l`

# Get first interferogram dates
foreach line (`head -1 $slist`)
    set inp = $dir2"/"$line"/"$ftype".grd"
    set beg = `echo $line | awk '{ print substr($1,1,8)}'`
    set end = `echo $line | awk '{ print substr($1,10,8)}'`
end

echo "Start date: $beg"
set prev=$beg

rm -f tmp.grd new.grd add.grd sum.grd

gmt grdmath $inp $inp SUB = dsum.grd
cp dsum.grd $ph0


# loop over all the acquisitions
set count = 0
foreach line (`cat $slist`)

    set date1 = `echo $line | awk '{ print substr($1,1,8)}'`
    set date2 = `echo $line | awk '{ print substr($1,10,8)}'`
    set count = `echo $count | awk '{print $1+1}'`

    if (`echo "$count > 1.1" | bc`) then

        #  echo $date1"_"$date2
        set inps = $dir1"/"$date1"_"$date2"/"$ftype".grd"
        set inps2 = $dir1"/"$prev"_"$date1"/"$ftype".grd"

        echo "working on $line..."

        if ( -e $inps) then
            gmt grdmath $inps2 $inps ADD $ph0 SUB $ph0 SUB = tmp.grd

            # wrap:
            gmt grdmath tmp.grd PI ADD 2 PI MUL MOD PI SUB = sum.grd
            echo "$inps2 + $inps"

        else
            echo "can't open $inps"
            exit 1
        endif

        set new = $dir1"/"$prev"_"$date2"/"$ftype".grd"
        set mean_phase1=`gmt grdinfo sum.grd -L | grep rms| awk '{printf "%.7f", $3}'`
        set mean_phase2=`gmt grdinfo $new -L | grep rms| awk '{printf "%.7f", $3}'`
        set dif=`echo "$mean_phase1 - $mean_phase2" |bc`
        set fc=`echo "scale=1; $dif / $pi2" |bc -l|awk '{printf "%.0f",$1}'`
        set fc2=`echo "scale=7; $fc * $pi2" |bc -l`
        set p1=`echo "scale=7; $mean_phase1" | bc -l`
        set p2=`echo "scale=7; $mean_phase2" | bc -l`

        if (`echo "$mean_phase1 < 0"|bc`) then
            set p1=`echo "0 - $mean_phase1 " |bc -l`
        endif

        if (`echo "$mean_phase2 < 0"|bc`) then
            set p2=`echo "0 - $mean_phase2 " |bc -l`
        endif

        if (`echo "$fc2 != 0"|bc`) then
            echo "n12+n23: $mean_phase1 |$p1| -n13: $mean_phase2 |$p2| fc: $fc $fc2"
            echo "$new"
            echo "$inps"

            if (`echo "$p1 > $p2"|bc`) then
                gmt grdmath sum.grd $fc2 SUB = tmp.grd
                mv tmp.grd sum.grd
                gmt grdmath $inps $fc2 SUB = tmp.grd
                mv $inps $inps.orig2
                mv tmp.grd $inps
                echo $fc2  >> $dir1"/"$date1"_"$date2"/adjust"

            else
                gmt grdmath $new $fc2 ADD = tmp.grd
                mv $new $new.orig2
                mv tmp.grd $new
                echo $fc2  >> $dir1"/"$prev"_"$date2"/adjust"

            endif
        endif

        if ( -e $mskf) then
            gmt grdmath sum.grd $new SUB $ph0 ADD $mskf MUL = $ftype"_"$beg"_"$date2"_diff".grd
        else
            gmt grdmath sum.grd $new SUB $ph0 ADD = $ftype"_"$beg"_"$date2"_diff".grd
        endif

        # wrap: 
        gmt grdmath $ftype"_"$beg"_"$date2"_diff".grd PI ADD 2 PI MUL MOD PI SUB = tmp.grd
        mv tmp.grd $ftype"_"$beg"_"$date2"_diff".grd

        cp $ftype"_"$beg"_"$date2"_diff".grd dsum.grd

        set rms_phase = `gmt grdinfo $ftype"_"$beg"_"$date2"_diff.grd" -L | grep rms| awk '{printf "%.7f", $7}'`
        set mean_phase = `gmt grdinfo $ftype"_"$beg"_"$date2"_diff.grd" -L | grep rms| awk '{printf "%.7f", $3}'`
        echo $date2 $rms_phase $mean_phase >> $out

    endif
    set prev = $date1
end

echo "Total # of igrams in the loop: $count"
