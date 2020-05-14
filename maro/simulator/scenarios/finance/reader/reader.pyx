#cython: language_level=3

import time
from enum import IntEnum
from math import floor, ceil
from cython cimport view

from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t


FALSE = 0
TRUE = 1


# since we will specified include folder, so we do not need to use relative path
cdef extern from "converter.c":
    ctypedef struct stock_t:
        uint8_t is_valid
        float opening_price
        float closing_price
        float pre_closing_price
        float highest_price
        float lowest_price
        float up_down_amount
        float up_down_rate
        float turnover_rate
        float trade_amount
        float total_market_capitalization
        float circulation_market_capitalization
        uint32_t trade_volume
        uint32_t trade_num
        uint32_t code
        uint64_t time
        float daily_return
 
    ctypedef struct meta_t:
        char header[4]
        uint8_t dtype
        uint8_t version
        uint16_t item_size
        uint32_t id
        uint64_t start_time
        uint64_t end_time

    ctypedef struct finreader_t:
        int fd
        int size
        void *addr
        meta_t meta
        int start
        int cur_index
        uint8_t dtype
        void *data

    uint8_t init_reader(const char *path, finreader_t *reader, uint8_t dtype)
    void release_reader(finreader_t *reader)
    uint8_t next_item(finreader_t *reader)
    uint8_t jump_to(finreader_t *reader, int index)
    void reset_reader(finreader_t *reader)

class FinanceDataType:
    STOCK = 1
    # FUTURES = 2


cdef fix_price(float val):
    """Fix the float price value"""
    return round(val, 2)

 
cdef class Stock:
    """Stock item info.
    NOTE: this object only used to hold result from binary for query, do not keep the reference"""
    cdef:
        stock_t *stock

    def __cinit__(self):
        pass

    cdef fill(self, void *data):
        self.stock = <stock_t*>data
 
    @property
    def is_valid(self):
        return False if not self.stock else self.stock.is_valid == 0

    @property
    def code(self):
        return None if not self.stock else self.stock.code

    @property
    def time(self):
        return None if not self.stock else self.stock.time

    @property
    def opening_price(self):
        return None if not self.stock else fix_price(self.stock.opening_price)

    @property
    def pre_closing_price(self):
        return  None if not self.stock else fix_price(self.stock.pre_closing_price)

    @property
    def closing_price(self):
        return  None if not self.stock else fix_price(self.stock.closing_price)

    @property
    def highest_price(self):
        return  None if not self.stock else fix_price(self.stock.highest_price)

    @property
    def lowest_price(self):
        return  None if not self.stock else fix_price(self.stock.lowest_price)

    @property
    def up_down_amount(self):
        return  None if not self.stock else self.stock.up_down_amount

    @property
    def up_down_rate(self):
        return  None if not self.stock else self.stock.up_down_rate

    @property
    def turnover_rate(self):
        return  None if not self.stock else self.stock.turnover_rate

    @property
    def trade_amount(self):
        return  None if not self.stock else self.stock.trade_amount

    @property
    def total_market_capitalization(self):
        return  None if not self.stock else self.stock.total_market_capitalization

    @property
    def circulation_market_capitalization(self):
        return  None if not self.stock else self.stock.circulation_market_capitalization

    @property
    def trade_volume(self):
        return  None if not self.stock else fix_price(self.stock.trade_volume)

    @property
    def trade_num(self):
        return  None if not self.stock else self.stock.trade_num
 
    @property
    def daily_return(self):
        return None if not self.stock else self.stock.daily_return
 
    def __repr__(self):
        return f"Stock (is_valid: {self.is_valid}, code: {self.code}, time: {self.time}, open price: {self.opening_price}, close price: {self.closing_price}, daily_return: {self.daily_return})"

cdef class FinanceReader:
    """Binary reader for finance scenario"""
    cdef:
        finreader_t reader
        uint8_t dtype
        char *path

        Stock stock

        # start index
        int _max_tick

        # used to hold padding number when the file start time greater than specified beginning time
        int _padding_days
        int _cur_padding

    def __cinit__(self, uint8_t dtype, char *path, start_tick: int, max_tick: int, beginning_time_stamp: int):
        self.dtype = dtype
        self.path = path
        self._max_tick = max_tick
        self._cur_padding = 0
        self._padding_days = 0
        self.stock = Stock()

        if FALSE == init_reader(self.path, &self.reader, dtype):
            raise BaseException("Fail to initialize Finance reader")

        # NOTE: we have to set start and num correctly to get item
        self.reader.start = 0

        day_seconds = 24 * 60 * 60

        if self.reader.meta.start_time > beginning_time_stamp:
            self._padding_days = ceil((self.reader.meta.start_time - beginning_time_stamp) / day_seconds)
            self._padding_days -= start_tick
            self._cur_padding = self._padding_days

        elif self.reader.meta.start_time < beginning_time_stamp:
            self.reader.start = floor((beginning_time_stamp - self.reader.meta.start_time) / day_seconds)

        reset_reader(&self.reader)

        if max_tick <= 0:
            self._max_tick = self.reader.size + self._padding_days - self.reader.start

    @property
    def max_tick(self):
        return self._max_tick

    @property
    def size(self):
        """Item count"""
        return self.reader.size
 
    @property
    def data_type(self):
        return self.dtype

    @property
    def id(self):
        return self.reader.meta.id

    @property
    def start_time(self):
        return self.reader.meta.start_time

    @property
    def end_time(self):
        return self.reader.meta.end_time

    @property
    def version(self):
        return self.reader.meta.version

    def next_item(self):
        if self._cur_padding > 0:
            self._cur_padding -= 1

            return None

        if FALSE == next_item(&self.reader):
            return None

        if self.dtype == FinanceDataType.STOCK:
            self.stock.fill(self.reader.data)

            return self.stock

    def __iter__(self):
        return self

    def __next__(self):
        item = self.next_item()

        if item is None:
            raise StopIteration
        else:
            return item

    def reset(self):
        reset_reader(&self.reader)
        self._cur_padding = self._padding_days
        # init_reader(self.path, &self.reader, self.dtype)

    def __dealloc__(self):
        release_reader(&self.reader)

    def __repr__(self):
        return f"FinanceReader (data type: {self.data_type}, code: {self.id}, count: {self.size}, start time: {self.start_time}, end time: {self.end_time})";