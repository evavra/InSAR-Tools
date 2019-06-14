import codecs
import os


def createCSV(stationList):
    for station in stationList:
        txt2csv(station)


def txt2csv(filename):
    # Created on 06/06/2019
    # Method for converting UNR-formatted GPS data textfiles to CSV format.

    # 1. Replace all spaces in original text file with commas
    f = codecs.open(filename + '.txt', encoding='utf-8')
    textFile = f.read()

    step1 = textFile.replace(' ', ',')
    temp = codecs.open('tempFile', 'w')

    temp.write(step1)
    f.close()

    # 2. Replace triple-commas with a single comma
    f2 = codecs.open('tempFile', encoding='utf-8')
    newFile = f2.read()

    step2 = newFile.replace(',,,', ',')
    temp2 = codecs.open('tempFile2', 'w')

    temp2.write(step2)
    f2.close()

    # 3. Replace double-commas with a single comma
    f3 = codecs.open('tempFile2', encoding='utf-8')
    newFile2 = f3.read()

    step3 = newFile2.replace(',,', ',')
    final = codecs.open(filename + '.csv', 'w')

    final.write(step3)

    f3.close()

    os.remove('tempFile')
    os.remove('tempFile2')


def createVector(filename):
    # Created on 06/06/2019
    # Method for converting UNR-formatted GPS data textfiles to CSV format.

    # 1. Replace all spaces in original text file with commas
    f = codecs.open(filename + '.txt', encoding='utf-8')
    textFile = f.read()

    temp = textFile.replace(' ', '\n')
    newFile = codecs.open(filename + '-new.txt', 'w')

    newFile.write(temp)
    f.close()
