# Sentinel Utilities

import subprocess
import os
import sys
import glob
import datetime as dt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import collections
import netcdf_read_write


def get_all_xml_names(directory, polarization, swath):
    pathname1 = directory + "/*-vv-*-00" + swath + ".xml"
    pathname2 = directory + "/*-vv-*-00" + str(int(swath) + 3) + ".xml"
    list_of_images_temp = glob.glob(pathname1) + glob.glob(pathname2)
    list_of_images = []
    for item in list_of_images_temp:
        list_of_images.append(item[:])
    return list_of_images


def get_manifest_safe_names(directory):
    mansafe = glob.glob(directory + '/manifest.safe')
    return mansafe


def get_all_tiff_names(directory, polarization, swath):
    pathname1 = directory + "/*-vv-*-00" + swath + ".tiff"
    pathname2 = directory + "/*-vv-*-00" + str(int(swath) + 3) + ".tiff"
    list_of_images_temp = glob.glob(pathname1) + glob.glob(pathname2)
    list_of_images = []
    for item in list_of_images_temp:
        list_of_images.append(item[:])
    return list_of_images


def get_previous_and_following_day(datestring):
    """ This is a function that takes a date like 20160827 and generates
    [20160826, 20160828]: the day before and the day after the date in question. """
    year = int(datestring[0:4]);
    month = int(datestring[4:6]);
    day = int(datestring[6:8]);
    mydate = dt.date(year, month, day)
    tomorrow = mydate + dt.timedelta(days=1)
    yesterday = mydate - dt.timedelta(days=1)
    previous_day = pad_string_zeros(yesterday.year) + pad_string_zeros(yesterday.month) + pad_string_zeros(yesterday.day)
    following_day = pad_string_zeros(tomorrow.year) + pad_string_zeros(tomorrow.month) + pad_string_zeros(tomorrow.day)
    return [previous_day, following_day]


def get_date_from_xml(xml_name):
    """
    xml file has name like s1a-iw1-slc-vv-20150121t134413-20150121t134424-004270-005317-001.xml
    We want to return 20150121.
    """
    xml_name = xml_name.split('/')[-1]
    mydate = xml_name[15:23];
    return mydate


def get_sat_from_xml(xml_name):
    xml_name = xml_name.split('/')[-1]
    sat = xml_name[0:3];
    return sat


def pad_string_zeros(num):
    if num < 10:
        numstring = "0" + str(num)
    else:
        numstring = str(num)
    return numstring


def get_eof_from_date_sat(mydate, sat, eof_dir):
    """ This returns something like S1A_OPER_AUX_POEORB_OPOD_20160930T122957_V20160909T225943_20160911T005943.EOF.
        It takes something like 20171204, s1a, eof_dir
    """
    [previous_day, following_day] = get_previous_and_following_day(mydate)
    eof_name = glob.glob(eof_dir + "/" + sat.upper() + "*" + previous_day + "*" + following_day + "*.EOF")
    if eof_name == []:
        print("ERROR: did not find any EOF files matching the pattern " + eof_dir + "/" + sat.upper() + "*" + previous_day + "*" + following_day + "*.EOF");
        print("Exiting...")
        sys.exit(1)
    else:
        eof_name = eof_name[0]
    return eof_name


def glob_intf_computed():
    full_names = glob.glob("intf_all/*")
    intf_computed = []
    for item in full_names:
        intf_computed.append(item[9:]);
    return intf_computed


def make_data_in(polarization, swath, master_date="00000000"):
    """
    data.in is a reference table that links the xml file with the correct orbit file.
    """
    list_of_images = get_all_xml_names("raw_orig", polarization, swath)
    outfile = open("data.in", 'w')
    if master_date == "00000000":
        for item in list_of_images:
            item = item.split("/")[-1]  # getting rid of the directory
            mydate = get_date_from_xml(item)
            sat = get_sat_from_xml(item)
            eof_name = get_eof_from_date_sat(mydate, sat, "raw_orig")
            outfile.write(item[:-4] + ":" + eof_name.split("/")[-1] + "\n");
    else:
        # write the master date first.
        for item in list_of_images:
            mydate = get_date_from_xml(item)
            if mydate == master_date:
                item = item.split("/")[-1]  # getting rid of the directory
                mydate = get_date_from_xml(item)
                sat = get_sat_from_xml(item)
                eof_name = get_eof_from_date_sat(mydate, sat, "raw_orig")
                outfile.write(item[:-4] + ":" + eof_name.split("/")[-1] + "\n");
        # then write the other dates.
        for item in list_of_images:
            mydate = get_date_from_xml(item)
            if mydate != master_date:
                item = item.split("/")[-1]  # getting rid of the directory
                mydate = get_date_from_xml(item)
                sat = get_sat_from_xml(item)
                eof_name = get_eof_from_date_sat(mydate, sat, "raw_orig")
                outfile.write(item[:-4] + ":" + eof_name.split("/")[-1] + "\n");
    outfile.close()
    print("data.in successfully printed.")
    return


def read_baseline_table(baselinefilename):
    baselineFile = np.genfromtxt(baselinefilename, dtype=str)
    stems = baselineFile[:, 0].astype(str);
    times = baselineFile[:, 1].astype(float);
    missiondays = baselineFile[:, 2].astype(str);
    baselines = baselineFile[:, 4].astype(float);
    return [stems, times, baselines, missiondays]


def read_intf_table(tablefilename):
    tablefile = np.genfromtxt(tablefilename, dtype=str)
    intf_all = tablefile[:].astype(str);
    return intf_all


def write_intf_table(intf_all, tablefilename):
    ofile = open(tablefilename, 'w')
    for i in intf_all:
        ofile.write("%s\n" % i)
    ofile.close()
    return

# after running the baseline calculation from the first pre_proc_batch, choose a new master that is close to the median baseline and timespan.


def choose_master_image():
    # load baseline table
    baselineFile = np.genfromtxt('raw/baseline_table.dat', dtype=str)
    time = baselineFile[:, 1].astype(float)
    baseline = baselineFile[:, 4].astype(float)
    shortform_names = baselineFile[:, 0].astype(str);

    # GMTSAR (currently) guarantees that this file has the same order of lines as baseline_table.dat.
    dataDotIn = np.genfromtxt('raw/data.in', dtype='str').tolist()
    print(dataDotIn)

    # calculate shortest distance from median to scenes
    consider_time = True
    if consider_time:
        time_baseline_scale = 1  # arbitrary scaling factor, units of (meters/day)
        sceneDistance = np.sqrt(((time - np.median(time)) / time_baseline_scale)**2 + (baseline - np.median(baseline))**2)
    else:
        sceneDistance = np.sqrt((baseline - np.median(baseline))**2)

    minID = np.argmin(sceneDistance)
    masterID = dataDotIn[minID]

    # put masterId in the first line of data.in
    dataDotIn.pop(dataDotIn.index(masterID))
    dataDotIn.insert(0, masterID)
    master_shortform = shortform_names[minID]  # because GMTSAR initially puts the baseline_table and data.in in the same order.

    os.rename('raw/data.in', 'raw/data.in.old')
    np.savetxt('raw/data.in', dataDotIn, fmt='%s')
    np.savetxt('data.in', dataDotIn, fmt='%s')
    return master_shortform


def write_super_master_batch_config(masterid):
    ifile = open('batch.config', 'r')
    ofile = open('batch.config.new', 'w')
    for line in ifile:
        if 'master_image' in line:
            ofile.write('master_image = ' + masterid + '\n')
        else:
            ofile.write(line)
    ifile.close()
    ofile.close()
    subprocess.call(['mv', 'batch.config.new', 'batch.config'], shell=False)
    print("Writing master_image into batch.config")
    return


def write_ordered_unwrapping(numproc, sh_file, config_file):
    [stem1, stem2, mean_corr] = read_corr_results("corr_results.txt")

    stem1_ordered = [x for y, x in sorted(zip(mean_corr, stem1), reverse=True)]
    stem2_ordered = [x for y, x in sorted(zip(mean_corr, stem2), reverse=True)]
    mean_corr_ordered = sorted(mean_corr, reverse=True)

    outfile = open(sh_file, 'w')
    outfile.write("#!/bin/bash\n")
    outfile.write("# Script to batch unwrap Sentinel-1 TOPS mode data sets.\n\n")
    outfile.write("rm intf?.in\n")
    for i, item in enumerate(stem1_ordered):
        outfile.write('echo "' + stem1_ordered[i] + ":" + stem2_ordered[i] + '" >> intf' + str(np.mod(i, numproc)) + '.in\n');
    outfile.write("\n# Unwrap the interferograms.\n\n")
    outfile.write("ls intf?.in | parallel --eta 'unwrap_mod.csh {} " + config_file + "'\n\n\n")
    outfile.close()

    return


def write_unordered_unwrapping(numproc, sh_file, config_file):
    infile = 'intf_record.in'
    intfs = []
    for line in open(infile):
        intfs.append(line[0:-1]);
    outfile = open(sh_file, 'w')
    outfile.write("#!/bin/bash\n")
    outfile.write("# Script to batch unwrap Sentinel-1 TOPS mode data sets.\n\n")
    outfile.write("rm intf?.in\n")
    for i, item in enumerate(intfs):
        outfile.write('echo "' + item + '" >> intf' + str(np.mod(i, numproc)) + '.in\n')
        # outfile.write("echo S1A20180106_ALL_F1:S1A20180118_ALL_F1 >> intf0.in\n"); break;
    outfile.write("\n# Unwrap the interferograms.\n\n")
    outfile.write("ls intf?.in | parallel --eta 'unwrap_mod.csh {} " + config_file + "'\n\n\n")
    outfile.close()

    return


def read_corr_results(corr_file):
    stem1 = []
    stem2 = []
    mean_corr = []
    ifile = open(corr_file, 'r')
    for line in ifile:
        temp = line.split()
        if len(temp) == 4:
            stem1.append(temp[1].split('.')[0]);  # format: S1A20171215
            stem2.append(temp[2].split('.')[0])
            mean_corr.append(float(temp[3]))
    return [stem1, stem2, mean_corr]


def remove_nans_array(myarray):
    numarray = []
    for i in range(len(myarray)):
        if ~np.isnan(myarray[i]):
            numarray.append(myarray[i][0])
    return numarray


def get_small_baseline_subsets(stems, tbaseline, xbaseline, tbaseline_max, xbaseline_max, startdate='', enddate=''):
    """ Grab all the pairs that are below the critical baselines in space and time.
    Return format is a list of strings like 'S1A20150310_ALL_F1:S1A20150403_ALL_F1'.
    You can adjust this if you have specific processing needs.
    """
    nacq = len(stems)
    if len(startdate) > 1:
        startdate_dt = dt.datetime.strptime(startdate, "%Y%j")
    else:
        startdate_dt = dt.datetime.strptime("2000001", "%Y%j")  # if there's no startdate, take everything.
    if len(enddate) > 1:
        enddate_dt = dt.datetime.strptime(enddate, "%Y%j")
    else:
        enddate_dt = dt.datetime.strptime("2100321", "%Y%j")  # if there's no enddate, take everything.
    intf_pairs = []
    datetimearray = []
    for k in tbaseline:
        datetimearray.append(dt.datetime.strptime(str(int(k) + 1), "%Y%j"))  # convert to datetime arrays.
    print(datetimearray)
    for i in range(0, nacq):
        for j in range(i + 1, nacq):
            dtdelta = datetimearray[i] - datetimearray[j]
            dtdeltadays = dtdelta.days  # how many days exist between the two acquisitions?
            if datetimearray[i] > startdate_dt and datetimearray[j] > startdate_dt:
                if datetimearray[i] < enddate_dt and datetimearray[j] < enddate_dt:
                    if abs(dtdeltadays) < tbaseline_max:
                        if abs(xbaseline[i] - xbaseline[j]) < xbaseline_max:
                            img1_stem = stems[i]
                            img2_stem = stems[j]
                            img1_time = int(img1_stem[3:11]);
                            img2_time = int(img2_stem[3:11]);
                            if img1_time < img2_time:  # if the images are listed in chronological order
                                intf_pairs.append(stems[i] + ":" + stems[j]);
                            else:                    # if the images are in reverse chronological order
                                intf_pairs.append(stems[j] + ":" + stems[i]);
                        else:
                            print("WARNING: %s:%s rejected due to large perpendicular baseline of %f m." % (stems[i], stems[j], abs(xbaseline[i] - xbaseline[j])));
    print("SBAS Pairs: Returning " + str(len(intf_pairs)) + " of " + str(nacq * (nacq - 1) / 2) + " possible interferograms to compute. ")
    # The total number of pairs is (n*n-1)/2.  How many of them fit our small baseline criterion?
    return intf_pairs


def get_chain_subsets(stems, tbaseline, xbaseline, bypass):
    # goal: order tbaselines ascending order. Then just take adjacent stems as the intf pairs.
    intf_pairs = []
    bypass_items = bypass.split("/")
    sorted_stems = [x for _, x in sorted(zip(tbaseline, stems))]  # sort by increasing t value
    for i in range(len(sorted_stems) - 1):
        intf_pairs.append(sorted_stems[i] + ':' + sorted_stems[i + 1]);
        if i > 1 and sorted_stems[i][3:11] in bypass_items:
            intf_pairs.append(sorted_stems[i - 1] + ':' + sorted_stems[i + 1])
    print("Connected Chain: Returning " + str(len(intf_pairs)) + " interferograms to compute. ")
    return intf_pairs


def get_manual_chain(stems, tbaseline, tbaseline_max, force_chain_images):
    # The point of this is to manually force the SBAS algorithm to connect adjacent scenes,
    # even if they're technically over the time limit used in the SBAS algorithm.
    # Force_chain_images is an array of images (first images) that were rejected in SBAS due to large perpendicular baseline
    # But we want to force the interferograms to be made anyway, regardless of perpendicular baseline.
    intf_pairs = []
    sorted_stems = [x for _, x in sorted(zip(tbaseline, stems))]  # sort by increasing t value
    sorted_tbaseline = sorted(tbaseline)
    sorted_datetimes = [dt.datetime.strptime(str(int(k) + 1), "%Y%j") for k in sorted_tbaseline]
    for i in range(len(sorted_stems) - 1):
        deltadays = sorted_datetimes[i + 1] - sorted_datetimes[i]
        if deltadays.days >= tbaseline_max:
            intf_pairs.append(sorted_stems[i] + ':' + sorted_stems[i + 1]);
        for k in force_chain_images:  # if we have images that were rejected in SBAS due to large perpendicular baseline
            if k in sorted_stems[i]:
                intf_pairs.append(sorted_stems[i] + ':' + sorted_stems[i + 1]);
    print("Manual Chain: Returning " + str(len(intf_pairs)) + " interferograms to compute. ")
    # print(intf_pairs);
    return intf_pairs


def make_network_plot(intf_pairs, stems, tbaseline, xbaseline, plotname, baselinefile='raw/baseline_table.dat'):
    print("printing network plot")
    if len(stems) == 0 and len(xbaseline) == 0:
        [stems, times, baselines, missiondays] = read_baseline_table(baselinefile)
    if len(intf_pairs) == 0:
        print("Error! Cannot make network plot because there are no interferograms. ")
        sys.exit(1)
    xstart = []
    xend = []
    tstart = []
    tend = []

    # If there's a format like "S1A20160817_ALL_F2:S1A20160829_ALL_F2"
    if "S1" in intf_pairs[0]:
        for item in intf_pairs:
            scene1 = item[0:18];    # has some format like S1A20160817_ALL_F2
            scene2 = item[19:];     # has some format like S1A20160817_ALL_F2
            for x in range(len(stems)):
                if stems[x] == scene1:
                    xstart.append(xbaseline[x])
                    tstart.append(dt.datetime.strptime(str(int(tbaseline[x]) + 1), '%Y%j'))
                if stems[x] == scene2:
                    xend.append(xbaseline[x])
                    tend.append(dt.datetime.strptime(str(int(tbaseline[x]) + 1), '%Y%j'))

    # If there's a format like "2017089:2018101"....
    # WRITE THIS NEXT.
    if len(intf_pairs[0]) == 15:
        dtarray = []
        im1_dt = []
        im2_dt = []
        for i in range(len(times)):
            dtarray.append(dt.datetime.strptime(str(times[i])[0:7], '%Y%j'));

        # Make the list of datetimes for the images.
        for i in range(len(intf_pairs)):
            scene1 = intf_pairs[i][0:7];
            scene2 = intf_pairs[i][8:15];
            im1_dt.append(dt.datetime.strptime(scene1, '%Y%j'))
            im2_dt.append(dt.datetime.strptime(scene2, '%Y%j'))

        # Find the appropriate image pairs and baseline pairs
        for i in range(len(intf_pairs)):
            for x in range(len(dtarray)):
                if dtarray[x] == im1_dt[i]:
                    xstart.append(baselines[x])
                    tstart.append(dtarray[x])
                if dtarray[x] == im2_dt[i]:
                    xend.append(baselines[x])
                    tend.append(dtarray[x])

    plt.figure()
    plt.plot_date(tstart, xstart, '.b')
    plt.plot_date(tend, xend, '.b')
    for i in range(len(tstart)):
        plt.plot_date([tstart[i], tend[i]], [xstart[i], xend[i]], 'b')
    yrs_formatter = mdates.DateFormatter('%m-%y')
    plt.xlabel("Date")
    plt.gca().xaxis.set_major_formatter(yrs_formatter)
    plt.ylabel("Baseline (m)")
    plt.title("Network Geometry")
    plt.savefig(plotname)
    plt.close()
    print("finished printing network plot")
    return


def make_referenced_unwrapped(rowref, colref, prior_staging_directory, post_staging_directory):
    files = glob.glob(prior_staging_directory + "/*")
    print("Imposing reference pixel on %d files in %s; saving output in %s" % (len(files), prior_staging_directory, post_staging_directory))
    out_dir = post_staging_directory + "/"
    subprocess.call(['mkdir', '-p', out_dir], shell=False)

    for filename in files:
        individual_name = filename.split('/')[-1]
        print(individual_name)
        [xdata, ydata, zdata] = netcdf_read_write.read_grd_xyz(filename)
        # xdata is range/columns, ydata is azimuth/rows

        # Here we subtract the value of zdata[rowref][colref] to fix the refernece pixel.
        # referenced_zdata[rowref][colref]=0 by definition.
        referenced_zdata = np.zeros(np.shape(zdata))
        for i in range(len(ydata)):
            for j in range(len(xdata)):
                referenced_zdata[i][j] = zdata[i][j] - zdata[rowref][colref]
        print(referenced_zdata[rowref][colref])

        outname = out_dir + individual_name
        netcdf_read_write.produce_output_netcdf(xdata, ydata, referenced_zdata, 'phase', outname)
        netcdf_read_write.flip_if_necessary(outname)
    return


def implement_reference_pixel(data_all, rowref, colref):
    new_data_all = np.zeros(np.shape(data_all))
    zdim, rowdim, coldim = np.shape(data_all)
    for i in range(zdim):
        for j in range(rowdim):
            for k in range(coldim):
                new_data_all[i][j][k] = data_all[i][j][k] - data_all[i][rowref][colref]
    return new_data_all


#
# Reporting and defensive programming
#

def compare_intended_list_with_directory(intended_array, actual_array, errormsg):
    # This takes two lists of dates formatted as strings, such as ['2015321_2015345']
    # It prints out any members that exist in the intended_array but not the actual_array
    for item in intended_array:
        if item in actual_array:
            continue
        else:
            print("ERROR! %s expected, but not found in actual array." % item)
            print(errormsg)
    return


def check_intf_all_sanity():
    # Figure out whether all intended interferograms were made and unwrapped.

    print("Checking the progress of intf and unwrap steps. ")

    # The hardest part: Fix the differences between datetime formats in intf_record.in
    intended_intfs = np.genfromtxt('intf_record.in', dtype='str')
    intended_intfs = [i[3:11] + '_' + i[22:30] for i in intended_intfs];  # these come in formatted S1A20161206_ALL_F1:S1A20161230_ALL_F1
    date1 = [dt.datetime.strptime(i[0:8], "%Y%m%d") - dt.timedelta(days=1) for i in intended_intfs];
    date2 = [dt.datetime.strptime(i[9:17], "%Y%m%d") - dt.timedelta(days=1) for i in intended_intfs];
    date1 = [dt.datetime.strftime(i, "%Y%j") for i in date1]
    date2 = [dt.datetime.strftime(i, "%Y%j") for i in date2]
    intended_intfs = []
    for i in range(len(date1)):
        intended_intfs.append(date1[i] + '_' + date2[i])
    num_intended = len(set(intended_intfs))
    print("  intended interferograms: %d from intf_record.in" % len(intended_intfs));
    print("  unique intended interferograms: %d " % num_intended);

    # Check for duplicated items in intf_record.in (may exist);
    duplicates = [item for item, count in collections.Counter(intended_intfs).items() if count > 1]
    print("  duplicated elements in intf_record.in: ");
    print(duplicates)

    # Collect the actual intf_all directories
    actual_intfs = subprocess.check_output('ls -d intf_all/201*_201* ', shell=True)
    actual_intfs = actual_intfs.split('\n')
    actual_intfs = [value for value in actual_intfs if value != '']
    actual_intfs = [i.split('/')[-1] for i in actual_intfs]
    print("  actual interferograms: %d from intf_all directory " % len(actual_intfs));

    # Collect the unwrap.grd files
    actual_unwraps = subprocess.check_output('ls intf_all/unwrap.grd/*_unwrap.grd', shell=True)
    actual_unwraps = actual_unwraps.split('\n')
    actual_unwraps = [value for value in actual_unwraps if value != '']
    actual_unwraps = [i.split('/')[-1] for i in actual_unwraps]
    actual_unwraps = [i.split('_unwrap.grd')[0] for i in actual_unwraps]
    print('  unwrapped interferograms: %d from intf_all/unwrap.grd directory' % len(actual_unwraps))

    if num_intended == len(actual_intfs):
        print("Congratulations! All of your interferograms have been made. ")
    else:
        compare_intended_list_with_directory(intended_intfs, actual_intfs, 'is not made.')
    if num_intended == len(actual_unwraps):
        print("Congratulations! All of your interferograms have been unwrapped. ")
    else:
        compare_intended_list_with_directory(intended_intfs, actual_unwraps, 'is not unwrapped.')

    return


#
# Exceptions and Exception handling
#

# A special exception for when a directory is poorly situated, and is going to fail.
class Directory_error(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return(repr(self.value))
