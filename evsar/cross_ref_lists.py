#!/bin/python

import sys

def main():
    """
    Check to see which items in 'new' are not in 'reference'

    Usage: python cross_ref_lists.py new reference
    """

    if len(sys.argv) != 3:
        print(main.__doc__)
        sys.exit()

    new = read_file(sys.argv[1])
    old = read_file(sys.argv[2])

    check_items(new, reference)

def read_file(filename):
    file_list = []
    with open(filename, 'r') as f:
        for line in f:
            append(line)

    return file_list

def check_items(new, reference):

    missing = []
    for item in new:
        if item not in reference:
            missing.append(item)

    print(missing)

    with open("missing_list_items.txt","w") as newFile:
        for line in missing:
            newFile.write("%s\n" % line)

    return missing

if __name__ == '__main__':
    main()