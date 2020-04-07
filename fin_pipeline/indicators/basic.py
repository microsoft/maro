import numpy
import pandas


def simple_moving_average(org_data, target_column, window):
    x = org_data[target_column]
    s = pandas.Series(x)

    return s.rolling(window).mean().to_numpy()


def exponential_weighted_moving_average(org_data, target_column, window, decay_rate):
    decay_array = _gen_decay(window, decay_rate)
    x = org_data[target_column]
    s = pandas.Series(x)
    return s.rolling(window).apply(lambda input_array: _cal_windowed_moving_average(input_array, decay_array))

def linear_weighted_moving_average(org_data, target_column):
    ndays = len(org_data)
    x = org_data[target_column]
    s = pandas.Series(x)
    return  s.rolling(ndays, min_periods=1).apply(lambda input_array: _cal_linear_moving_average(input_array, _gen_linear_decay_array(input_array)))

def returns(org_data, target_column, window=2):
    x = org_data[target_column]
    s = pandas.Series(x)
    return (s / s.shift(window-1) - 1).to_numpy()

def percent_change(org_data, target_column, window=2):
    x = org_data[target_column]
    s = pandas.Series(x)
    return ((s / s.shift(window-1) - 1).abs()*100).to_numpy()

def weighted_average(org_data, target_column, weight_column):
    x = org_data[target_column]
    s = pandas.Series(x)
    w = org_data[weight_column]
    sw = pandas.Series(w)
    return (s*sw).sum()/sw.sum()

def max_drawdown(org_data, target_column):
    x = org_data[target_column]
    s = pandas.Series(x)
    window = len(s)
    Roll_Max = s.rolling(window, min_periods=1).max()
    Daily_Drawdown = s/Roll_Max - 1.0
    Max_Daily_Drawdown = Daily_Drawdown.rolling(window, min_periods=1).min()
    return Max_Daily_Drawdown.to_numpy()

def _gen_dttype(target_obj):
    names = []
    formats = []
    for key in target_obj.keys():
        names.append(key)
        format =  f"U50"
        if _check_int(target_obj[key]):
            format = 'i8'
        elif _check_float(target_obj[key]):
            format = 'f8'
        formats.append(format)
    return {'names': tuple(names), 'formats': tuple(formats)}


def _check_int(s):
    s = str(s)
    if s[0] in ('-', '+'):
        return s[1:].isdigit()
    return s.isdigit()


def _check_float(s):
    try:
        float_n = float(s)
    except ValueError:
        return False
    else:
        return True

def _gen_decay(window, decay_rate):
    return pandas.Series(numpy.full(window, decay_rate, numpy.dtype('float64')) ** numpy.arange(window , 0, -1))

def _cal_windowed_moving_average(input_array, window_array):
    ret = input_array.reset_index(drop=True).multiply(window_array)
    return ret.sum()/(window_array.sum())

def _gen_linear_decay_array(window_array):
    ndays = len(window_array)
    decay_array = pandas.Series(numpy.arange(1, ndays + 1, dtype=numpy.dtype('float64')))
    decay_array = decay_array / (ndays * (ndays + 1) / 2)
    return decay_array

def _cal_linear_moving_average(input_array, window_array):
    ret = input_array.reset_index(drop=True).multiply(window_array)
    return ret.sum()


if __name__ == "__main__":
    #read json file
    import json
    import os
    import sys
    datafile = sys.argv[1]
    if os.path.exists(datafile):
        with open(datafile, mode="r", encoding="utf-8") as json_file:
            data = json.load(json_file)
            adata = []
            for key in data.keys():
                if key not in ['latest', 'oldest']:
                    adata.append(data[key])
            #convert to numpy array
            if len(adata) != 0:
                dtype = _gen_dttype(adata[0])
                values = [tuple(d[name] for name in dtype['names']) for d in adata]
                npdata = numpy.array(values, dtype=dtype)
                print(npdata)
                print(npdata.dtype.names)
                #calculate moving average
                sma = simple_moving_average(npdata, 'closing_price', 3)
                print('closing_price')
                print(npdata['closing_price'])
                print(sma)
                ret = returns(npdata, 'closing_price', 3)
                print(ret)
                pecChange = percent_change(npdata, 'closing_price', 3)
                print(pecChange)
                wa = weighted_average(npdata, 'closing_price', 'trade_volume')
                print(wa)
                max_dw = max_drawdown(npdata,'closing_price')
                print(max_dw)
                ewma = exponential_weighted_moving_average(npdata, 'closing_price', 3, 0.9)
                print(ewma)
                lwma = linear_weighted_moving_average(npdata, 'closing_price')
                print(lwma)
