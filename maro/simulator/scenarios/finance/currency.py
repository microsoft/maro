from enum import Enum
from os import path
from collections import OrderedDict

import pandas as pd
import json


CurrencyType = Enum(
    value='CurrencyType',
    # currency types from https://www.ofx.com/en-us/forex-news/historical-exchange-rates/
    names=[
        ('ARS', "Argentine Peso"),
        ('AUD', "Australian Dollar"),
        ('AZN', "Azerbaijani New Manat"),
        ('BGN', "Bulgarian Lev"),
        ('BHD', "Bahraini Dinar"),
        ('BND', "Brunei Dollar"),
        ('BRL', "Brazilian Real"),
        ('CAD', "Canadian Dollar"),
        ('CHF', "Swiss Franc"),
        ('CLP', "Chilean Peso"),
        ('CNH', "Chinese Renminbi Off-Shore"),
        ('CNY', "Chinese Yuan"),
        ('CZK', "Czech Koruna"),
        ('DKK', "Danish Kroner"),
        ('EGP', "Egyptian Pound"),
        ('EUR', "Euro"),
        ('FJD', "Fiji Dollar"),
        ('GBP', "British Pound"),
        ('HKD', "Hong Kong Dollar"),
        ('HUF', "Hungarian Forint"),
        ('IDR', "Indonesian Rupiah"),
        ('ILS', "Israeli New Shekel"),
        ('INR', "Indian Rupee"),
        ('JPY', "Japanese Yen"),
        ('KRW', "Korean Won"),
        ('KWD', "Kuwaiti Dinar"),
        ('LKR', "Sri Lanka Rupee"),
        ('MAD', "Moroccan Dirham"),
        ('MGA', "Malagasy Ariary"),
        ('MXN', "Mexican Peso"),
        ('MYR', "Malaysian Ringgit"),
        ('NOK', "Norwegian Kroner"),
        ('NZD', "New Zealand Dollar"),
        ('OMR', "Omani Rial"),
        ('PEN', "Peruvian Nuevo Sol"),
        ('PGK', "Papua New Guinea Kina"),
        ('PHP', "Philippine Peso"),
        ('PKR', "Pakistani Rupee"),
        ('PLN', "Polish Zloty"),
        ('RUB', "Russian Rouble"),
        ('SAR', "Saudi Arabian Riyal"),
        ('SBD', "Solomon Islands Dollar"),
        ('SCR', "Seychelles Rupee"),
        ('SEK', "Swedish Krona"),
        ('SGD', "Singapore Dollar"),
        ('THB', "Thai Baht"),
        ('TOP', "Tonga Pa`anga"),
        ('TRY', "Turkish Lira"),
        ('TWD', "New Taiwan Dollar"),
        ('TZS', "Tanzanian Shilling"),
        ('USD', "US Dollar"),
        ('VND', "Vietnamese Dong"),
        ('VUV', "Vanuatu Vatu"),
        ('WST', "Samoa Tala"),
        ('XOF', "CFA Franc (BCEAO)"),
        ('XPF', "Pacific Franc"),
        ('ZAR', "South African Rand")
    ]
)


class CurrencyExchange():
    data_path = ""  # root path of exchange data
    USD_transfer = False  # use USD to exchnge two other currency
    from_USD = False  # exchange from USD or to USD
    fallback_exchange = 1  # fix exchange rate if no exchange found
    exchange_data_path = ""  # exchange data file path
    exchange_data = None  # pd frame of exchange data
    exchange_to_USD = None  # exchange data of USD to Target currency
    exchange_from_USD = None  # exchange data of Source currency to USD
    from_currency = None  # source currency
    to_currency = None  # target currency
    init_time = None  # timestamp of tick 0

    def __init__(self, data_path: str, from_currency: str, to_currency: str, fallback_exchange: float = 1, init_time: pd._libs.tslibs.timestamps.Timestamp = pd.Timestamp(year=1970, month=1, day=1)):
        self.from_currency = CurrencyType[from_currency]
        self.to_currency = CurrencyType[to_currency]
        self.init_time = init_time
        if from_currency == to_currency:
            self.fallback_exchange = 1

        else:
            self.data_path = data_path
            if self.from_currency == CurrencyType.USD or self.to_currency == CurrencyType.USD:
                if self.from_currency == CurrencyType.USD:
                    self.exchange_data_path = f"currency_USD_to_{self.to_currency.name}.json"
                    self.from_USD = True
                else:
                    self.exchange_data_path = f"currency_USD_to_{self.from_currency.name}.json"
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
                self.exchange_to_USD = CurrencyExchange(data_path, from_currency, 'USD', fallback_exchange)
                self.exchange_from_USD = CurrencyExchange(data_path, 'USD', to_currency, fallback_exchange)

    def get_inter_bank_rate(self, date: pd._libs.tslibs.timestamps.Timestamp):
        if not self.USD_transfer:
            if self.exchange_data is not None:
                for i in range(14):
                    date_t = date + pd.to_timedelta(-i, unit='D')
                    exist_date = self.exchange_data[self.exchange_data["PointInTime"] == date_t]
                    if len(exist_date) > 0:
                        if self.from_USD:
                            return exist_date["InterbankRate"].values[0]
                        else:
                            return exist_date["InverseInterbankRate"].values[0]
                        break
            print(
                "Warning: exchange data not found, use fallback exchange. From",
                self.from_currency, "to", self.to_currency, self.fallback_exchange
            )
        else:
            if self.exchange_to_USD is not None and self.exchange_from_USD is not None:
                return self.exchange_to_USD.get_inter_bank_rate(date) * self.exchange_from_USD.get_inter_bank_rate(date)

        return self.fallback_exchange

    def get_inter_bank_rate_by_tick(self, tick: int):
        cur_date = self.init_time + pd.to_timedelta(tick, unit='D')
        return self.get_inter_bank_rate(cur_date)


class Singleton(object):
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Singleton, cls).__new__(cls)
        return cls._instance


class Exchanger(Singleton):
    _data_path = None
    _init_time = None
    _exchangers_dict = OrderedDict()

    @classmethod
    def get_exchanger(cls):
        return cls._instance

    def __init__(
            self, data_path: str,
            init_time: pd._libs.tslibs.timestamps.Timestamp = pd.Timestamp(year=1970, month=1, day=1)
        ):
        self._data_path = data_path
        self._init_time = init_time

    def exchange_from_by_date(
            self, from_currency: CurrencyType, to_currency: CurrencyType,
            amount: float, cur_date: pd._libs.tslibs.timestamps.Timestamp
        ):
        exchanger_key = from_currency.name + ':' + to_currency.name
        if exchanger_key not in self._exchangers_dict:
            self._exchangers_dict[exchanger_key] = CurrencyExchange(
                self._data_path, from_currency.name, to_currency.name, init_time=self._init_time
            )
        exchange_rate = self._exchangers_dict[exchanger_key].get_inter_bank_rate(cur_date)
        return exchange_rate * amount

    def exchange_from_by_tick(self, from_currency: CurrencyType, to_currency: CurrencyType, amount: float, tick: int):
        exchanger_key = from_currency.name + ':' + to_currency.name
        if exchanger_key not in self._exchangers_dict:
            self._exchangers_dict[exchanger_key] = CurrencyExchange(
                self._data_path, from_currency.name, to_currency.name, init_time=self._init_time
            )
        exchange_rate = self._exchangers_dict[exchanger_key].get_inter_bank_rate_by_tick(tick)
        return exchange_rate * amount

    def exchange_to_by_date(
            self, from_currency: CurrencyType, to_currency: CurrencyType,
            amount: float, cur_date: pd._libs.tslibs.timestamps.Timestamp
        ):
        exchanger_key = from_currency.name + ':' + to_currency.name
        if exchanger_key not in self._exchangers_dict:
            self._exchangers_dict[exchanger_key] = CurrencyExchange(
                self._data_path, from_currency.name, to_currency.name, init_time=self._init_time
            )
        exchange_rate = self._exchangers_dict[exchanger_key].get_inter_bank_rate(cur_date)
        return amount / exchange_rate

    def exchange_to_by_tick(self, from_currency: CurrencyType, to_currency: CurrencyType, amount: float, tick: int):
        exchanger_key = from_currency.name + ':' + to_currency.name
        if exchanger_key not in self._exchangers_dict:
            self._exchangers_dict[exchanger_key] = CurrencyExchange(
                self._data_path, from_currency.name, to_currency.name, init_time=self._init_time
            )
        exchange_rate = self._exchangers_dict[exchanger_key].get_inter_bank_rate_by_tick(tick)
        return amount / exchange_rate

if __name__ == "__main__":
    a = CurrencyExchange("/mnt/d/work/maro_git/maro/fin_pipeline/currency", 'CNY', 'SGD')
    test_date = pd.to_datetime('2020-05-11')
    print("test_date", type(test_date))
    exchange_rate = a.get_inter_bank_rate(test_date)
    print(exchange_rate)
    Exchanger("/mnt/d/work/maro_git/maro/fin_pipeline/currency")
    exchanger = Exchanger.get_exchanger()
    test_exchange = exchanger.exchange_to_by_date(CurrencyType.CNY, CurrencyType.USD, 10000, test_date)
    print("test_exchange", test_exchange)
