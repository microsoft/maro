# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import csv
import os
from os.path import dirname, join, realpath

OUTPUT_CSV_FOLDER = join(dirname(dirname(realpath(__file__))), "results")
os.makedirs(OUTPUT_CSV_FOLDER, exist_ok=True)
OUTPUT_CSV_PATH = join(OUTPUT_CSV_FOLDER, "baseline.csv")


def post_collect(info_list: list, ep: int, segment: int) -> None:
    with open(OUTPUT_CSV_PATH, "a") as fp:
        writer = csv.writer(fp, delimiter=' ')
        for info in info_list:
            writer.writerow([ep, info["sold"], info["demand"], info["sold/demand"]])
        # print the average env metric
        if len(info_list) > 1:
            metric_keys, num_envs = info_list[0].keys(), len(info_list)
            avg = {key: sum(info[key] for info in info_list) / num_envs for key in metric_keys}
            writer.writerow([ep, avg["sold"], avg["demand"], avg["sold/demand"]])


def post_evaluate(info_list: list, ep: int) -> None:
    with open(OUTPUT_CSV_PATH, "a") as fp:
        writer = csv.writer(fp, delimiter=' ')
        for info in info_list:
            writer.writerow([ep, info["sold"], info["demand"], info["sold/demand"]])
        # print the average env metric
        if len(info_list) > 1:
            metric_keys, num_envs = info_list[0].keys(), len(info_list)
            avg = {key: sum(info[key] for info in info_list) / num_envs for key in metric_keys}
            writer.writerow([ep, avg["sold"], avg["demand"], avg["sold/demand"]])
