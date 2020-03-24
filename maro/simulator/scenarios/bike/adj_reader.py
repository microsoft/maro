# read adj infor from file
import csv


def read_adj_info(file_path: str):
    adj = [] # index is the cell index value is the beighbors

    with open(file_path, "rt") as fp:
        reader = csv.reader(fp)

        for line in reader:
            adj.append([int(c) for c in line])

    return adj