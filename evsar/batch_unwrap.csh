#!/bin/csh -f
# 
# By Xiaohua XU, 03/12/2018
# Modified by Ellis Vavra, Mar. 2021
#
# Unwrap set of interferograms using GNU parallel
#
# IMPORTANT: put a script called unwrap_intf.csh in path
# e.g. 
#   cd $1
#   snaphu[_interp].csh 0.1 0 
#   cd ..
#

if ($#argv != 5) then
    echo ""
    echo "Unwrap batch of interferograms in parallel"
    echo ""
    echo "NOTE:" 
    echo " - GNU parallel must be installed to use batch_unwrap.csh"
    echo " - Subroutine unwrap_intf.csh must be located in path or current directory (example enclosed)"
    echo " - Script may be modified to send email to user. Modify address accordingly"
    echo ""
    echo "USAGE:" 
    echo "  batch_unwrap.csh intf_list intf_dir phase_file correlation_threshold n_cores"
    echo ""
    echo "INPUTS:"
    echo "  intf_list             - list of interferogram directories (YYYYMMDD_YYYYMMDD format)"
    echo "  intf_dir              - path to directory containing interferograms"
    echo "  phase_file            - filestem of interferogram files to unwrap (e.g. phase, phasefilt, phaseraw)"
    echo "  correlation_threshold - coherence threshhold for unwrapping"
    echo "  n_cores               - number of cores to use in parallel processing"
    echo ""
    exit
endif

# Get arguments
set intf_list             = $1
set intf_dir              = $2
set phase_file            = $3
set correlation_threshold = $4
set n_cores               = $5

# Track start time
set d1 = `date`

# Create individual unwrapping jobs
foreach line (`awk '{print $0}' $1`)
    # Assumes no phase discontinuities are present in time-series - modify '0' argument if so.
    echo "unwrap_intf.csh $line $correlation_threshold 0 > log_$line.txt" >> unwrap.cmd
end

# Initiate parallel batch of jobs
parallel --jobs $n_cores < batch_unwrap.cmd

# echo ""
# echo "Finished all unwrapping jobs..."
# echo ""

# Get end time
set d2 = `date`

rm -f batch_unwrap.cmd

# Send notice email to user
path = `pwd`
echo "Job started on $d1 and finished on $d2 at $path "| mail -s "Unwrapping finished" evavra@ucsd.edu


# --------------- Example unwrap_intf.csh ----------------
# #!/bin/csh -f

# if ($#argv != 2) then
#    echo ""
#    echo "Usage: unwrap_intf.csh dates unwrap_threshold"
#    echo ""
#    exit 1
# endif

# set dates = $1
# set threshold = $2
# cd $dates

# snaphu_interp.csh $threshold 0 
# # snaphu_interp.csh $threshold 50

# cd .. 
# --------------------------------------------------------