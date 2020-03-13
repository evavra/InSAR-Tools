#!/bin/csh -f
# unwrap.list - contains list of GMTSAR-formatted date1_date2 interferogram directories to unwrap

foreach line (`awk '{print $1}' intflist`)
    cd $line
    snaphu_interp.csh 0 0 0/20000/0/6000 
    mv unwrap.grd unwrap_0.15m.grd
    mv unwrap.pdf unwrap_0.15m.pdf
    mv unwrap.cpt unwrap_0.15m.cpt
    cd ..
end

