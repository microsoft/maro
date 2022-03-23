# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import csv
import os

OUTPUT_CSV_FOLDER = os.path.join(os.getcwd(), "logs", "supply_chain")
OUTPUT_CSV_PATH = os.path.join(OUTPUT_CSV_FOLDER, "result_baseline.csv")
if not os.path.exists(OUTPUT_CSV_FOLDER):
    os.makedirs(OUTPUT_CSV_FOLDER)


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
