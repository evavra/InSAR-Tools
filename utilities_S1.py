import numpy as np
import matplotlib.pyplot as plt
import datetime as dt


def get_date_from_xml(xml_name):
    """
    xml file has name like s1a-iw1-slc-vv-20150121t134413-20150121t134424-004270-005317-001.xml
    We want to return 20150121.
    """
    xml_name = xml_name.split('/')[-1]
    mydate = xml_name[15:23];
    return mydate


def getDatesFromList(fileName, fileType, numDates):
    """
    Retrive name from input .SAFE directory name
    ex:
    Input: S1A_IW_SLC__1SDV_20151209T135925_20151209T135955_008966_00CD9B_CA9F.SAFE
    Output: 20151209
    """

    if fileType == 'SAFE':
        print("Reading " + fileName)

        with open(fileName, 'r') as file:
            names = file.readlines()

        dates = []
        for line in names:
            dates.append(line[17:25])
            print(line[17:25])

        with open('SAFE-dates.txt', 'w') as newFile:
            for line in dates:
                newFile.write("%s\n" % line)

        print(dates)
        print('SAFE-dates.txt has been created')
        print()
        print()

        return dates

    elif fileType == "ASF":
        """
        Read in dates from Alaska Satellite Facility download file
        """
        print("Reading " + fileName)

        with open(fileName, 'r') as file:
            names = file.readlines()

        datesASF = []

        for i in range(78, 78 + numDates - 1):
            datesASF.append(names[i][80:88])
            print(names[i][80:88])

        with open('ASF-dates.txt', 'w') as newFile:
            for line in datesASF:
                newFile.write("%s\n" % line)

        print(datesASF)
        print('ASF-dates.txt has been created')

        return datesASF

    else:
        print("List must be SAFE directory names")


def plotSceneDates(dateList1, dateList2, xLabelList):
    dates1, dates2 = [], []

    for i in dateList1:
        dates1.append(dt.datetime.strptime(i, "%Y%m%d"))
        # print(dates1)

    for i in dateList2:
        dates2.append(dt.datetime.strptime(i, "%Y%m%d"))
        # print(dates2)

    # Make plot
    fig = plt.figure()

    plt.scatter(dates1, np.ones(len(dates1)) * 4.5, marker='.')
    plt.scatter(dates2, np.ones(len(dates2)) * 5.5, marker='.')
    plt.xlabel('Date')
    plt.yticks([4.5, 5.5], xLabelList)
    plt.ylim(0, 10)
    plt.grid(True)
    plt.show()
    # ax.set_aspect(aspect=0.2)


def checkItems(listIn, listSearch):
    listOut = []
    for item in listIn:
        if item not in listSearch:
            listOut.append(item)

    print(listOut)

    with open("Missing-items.txt","w") as newFile:
        for line in listOut:
            newFile.write("%s\n" % line)

    return listOut


if __name__ == "__main__":
    datesDownloaded = getDatesFromList('downloadedSAFE-new.txt', 'SAFE', 139)

    datesASF = getDatesFromList('all_S1_144_06142019.py', 'ASF', 151)

    #plotSceneDates(datesDownloaded, datesASF, ['Downloaded', 'All'])
    checkItems(datesASF, datesDownloaded)
