from enum import Enum
from os import path

import pandas as pd
import numpy as np
import json


class CurrencyType(Enum):
    ARS = "Argentine Peso"
    AUD = "Australian Dollar"
    AZN = "Azerbaijani New Manat"
    BGN = "Bulgarian Lev"
    BHD = "Bahraini Dinar"
    BND = "Brunei Dollar"
    BRL = "Brazilian Real"
    CAD = "Canadian Dollar"
    CHF = "Swiss Franc"
    CLP = "Chilean Peso"
    CNH = "Chinese Renminbi Off-Shore"
    CNY = "Chinese Yuan"
    CZK = "Czech Koruna"
    DKK = "Danish Kroner"
    EGP = "Egyptian Pound"
    EUR = "Euro"
    FJD = "Fiji Dollar"
    GBP = "British Pound"
    HKD = "Hong Kong Dollar"
    HUF = "Hungarian Forint"
    IDR = "Indonesian Rupiah"
    ILS = "Israeli New Shekel"
    INR = "Indian Rupee"
    JPY = "Japanese Yen"
    KRW = "Korean Won"
    KWD = "Kuwaiti Dinar"
    LKR = "Sri Lanka Rupee"
    MAD = "Moroccan Dirham"
    MGA = "Malagasy Ariary"
    MXN = "Mexican Peso"
    MYR = "Malaysian Ringgit"
    NOK = "Norwegian Kroner"
    NZD = "New Zealand Dollar"
    OMR = "Omani Rial"
    PEN = "Peruvian Nuevo Sol"
    PGK = "Papua New Guinea Kina"
    PHP = "Philippine Peso"
    PKR = "Pakistani Rupee"
    PLN = "Polish Zloty"
    RUB = "Russian Rouble"
    SAR = "Saudi Arabian Riyal"
    SBD = "Solomon Islands Dollar"
    SCR = "Seychelles Rupee"
    SEK = "Swedish Krona"
    SGD = "Singapore Dollar"
    THB = "Thai Baht"
    TOP = "Tonga Pa`anga"
    TRY = "Turkish Lira"
    TWD = "New Taiwan Dollar"
    TZS = "Tanzanian Shilling"
    USD = "US Dollar"
    VND = "Vietnamese Dong"
    VUV = "Vanuatu Vatu"
    WST = "Samoa Tala"
    XOF = "CFA Franc (BCEAO)"
    XPF = "Pacific Franc"
    ZAR = "South African Rand"


class CurrencyExchange():
    data_path = ""
    USD_transfer = False
    from_USD = False
    inited = False
    fallback_exchange = 1
    exchange_data_path = ""
    exchange_data = None
    exchange_func = None
    exchange_to_USD = None
    exchange_from_USD = None
    from_currency = None
    to_currency = None
    init_time = None

    def __init__(self, account_config: dict, from_currency: CurrencyType, to_currency: CurrencyType, fallback_exchange: float = 1, init_time: pd._libs.tslibs.timestamps.Timestamp = pd.Timestamp(year=1970, month=1, day=1)):
        self.from_currency = from_currency
        self.to_currency = to_currency
        self.init_time = init_time
        if from_currency == to_currency:
            self.fallback_exchange = 1

        else:
            if "exchange_path" in account_config:
                self.data_path = account_config["exchange_path"]
                if from_currency == CurrencyType.USD or to_currency == CurrencyType.USD:
                    if from_currency == CurrencyType.USD:
                        self.exchange_data_path = f"currency_USD_to_{to_currency.name}.json"
                        self.from_USD = True
                    else:
                        self.exchange_data_path = f"currency_USD_to_{from_currency.name}.json"
                        self.from_USD = False
                    file_path = path.join(self.data_path, self.exchange_data_path)
                    if path.exists(file_path):
                        with open(file_path, 'r') as f:
                            org_data = json.load(f)
                            pd_data = pd.DataFrame(org_data["HistoricalPoints"])
                            pd_data["PointInTime"] = pd.to_datetime(pd_data["PointInTime"], unit="ms")
                            self.exchange_data = pd_data
                else:
                    self.USD_transfer = True
                    self.exchange_to_USD = CurrencyExchange(account_config, from_currency, CurrencyType.USD, fallback_exchange)
                    self.exchange_from_USD = CurrencyExchange(account_config, CurrencyType.USD, to_currency, fallback_exchange)

    def get_inter_bank_rate(self, date: pd._libs.tslibs.timestamps.Timestamp):
        if not self.USD_transfer:
            if self.exchange_data is not None:
                print(type(self.exchange_data["PointInTime"][0]))
                for i in range(14):
                    date_t = date + pd.to_timedelta(-i, unit='D')
                    print("date_t", date_t)
                    exist_date = self.exchange_data[self.exchange_data["PointInTime"] == date_t]
                    if len(exist_date) > 0:
                        print("exist_date", exist_date)
                        if self.from_USD:
                            return exist_date["InterbankRate"].values[0]
                        else:
                            return exist_date["InverseInterbankRate"].values[0]
                        break
            print("Warning: exchange data not found, use fallback exchange. From", self.from_currency, "to", self.to_currency, self.fallback_exchange)
        else:
            if self.exchange_to_USD is not None and self.exchange_from_USD is not None:
                return self.exchange_to_USD.get_inter_bank_rate(date) * self.exchange_from_USD.get_inter_bank_rate(date)

        return self.fallback_exchange

    def get_inter_bank_rate_by_tick(self, tick: int):
        cur_date = self.init_time + pd.to_timedelta(tick, unit='D')
        return self.get_inter_bank_rate(cur_date)


if __name__ == "__main__":
    a = CurrencyExchange({"exchange_path": "/mnt/d/work/maro_git/maro/fin_pipeline/currency"}, CurrencyType.CNY, CurrencyType.SGD)
    test_date = pd.to_datetime('2020-05-11')
    print("test_date", type(test_date))
    exchange_rate = a.get_inter_bank_rate(test_date)
    print(exchange_rate)
