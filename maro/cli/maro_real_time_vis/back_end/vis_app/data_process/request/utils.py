import pandas as pd


def get_data_in_format(original_data):
    dataset = original_data["dataset"]
    column = original_data["columns"]
    dataheader = []
    for col_index in range(0, len(column)):
        dataheader.append(column[col_index]["name"])
    data_in_format = pd.DataFrame(dataset, columns=dataheader)
    return data_in_format


def get_input_range(start_tick, end_tick):
    i = start_tick
    input_range = "("
    while i < end_tick:
        if i == end_tick - 1:
            input_range = input_range + "'" + str(i) + "'" + ")"
        else:
            input_range = input_range + "'" + str(i) + "'" + ","
        i = i + 1
    return input_range
