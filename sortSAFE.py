import glob
import datetime as dt
import pandas as pd

# ------ sortSAFE.py ------------------------------------------------------------------
# - Method for organizing list of GMTSAR SAFE directories to be used in InSAR processing
# - Need to have directories organized by aquisition date rather than satellite ID (S1A/S1B) 
#   or polarization setting (1SSV/1SDV)
# - Run from 'data' directory containing raw SAR data to generate date-sorted list of files

# Get list of directory names
dirList = glob.glob('*SAFE')

# Pull dates from filenames
dates=[]

for name in dirList:
    dates.append(dt.datetime.strptime(name[17:25], '%Y%m%d'))

# Add to dataframe and sort by date
sortList = pd.DataFrame({'date':dates, 'name':dirList})

sortList = sortList.sort_values(by='date')

print(sortList)

# Save to new file called 'SAFE_filelist'
with open('SAFE_filelist', 'w') as newList:
    for i in range(len(dirList)):
        newList.write(sortList.iloc[(i, 1)] + '\n')

