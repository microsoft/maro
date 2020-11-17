import calendar

from dateutil.tz import UTC

from maro.data_lib.binary_converter import is_datetime
from maro.data_lib.binary_reader import unit_seconds


def get_cn_stock_data_tick(start_date: str) -> int:
    ret = None
    tzone = "Asia/Shanghai"
    default_start_dt = "1991-01-01"
    default_time_unit = "d"
    is_dt, dt = is_datetime(start_date, tzone)
    if is_dt:
        # convert into UTC, then utc timestamp
        # dt = dt.astimezone(UTC)
        _, start_dt = is_datetime(default_start_dt, tzone)
        dt_seconds = calendar.timegm(dt.timetuple())
        start_dt_seconds = calendar.timegm(start_dt.timetuple())
        delta_seconds = dt_seconds - start_dt_seconds
        seconds_per_unit = unit_seconds(default_time_unit)
        ret = int((delta_seconds) / seconds_per_unit)
    return ret

def get_stock_start_timestamp(start_date: str = "1991-01-01", tzone: str = "Asia/Shanghai") -> int:
    ret = None
    default_start_dt = "1970-01-01"
    default_start_tzone = "UTC"
    default_time_unit = "s"
    is_dt, dt = is_datetime(start_date, tzone)
    if is_dt:
        # convert into UTC, then utc timestamp
        # dt = dt.astimezone(UTC)
        _, start_dt = is_datetime(default_start_dt, default_start_tzone)
        # start_dt = start_dt.astimezone(UTC)
        dt_seconds = calendar.timegm(dt.timetuple())
        start_dt_seconds = calendar.timegm(start_dt.timetuple())
        delta_seconds = dt_seconds - start_dt_seconds
        seconds_per_unit = unit_seconds(default_time_unit)
        ret = int(delta_seconds / seconds_per_unit)
    return ret


if __name__ == "__main__":
    start_time = "2015-01-01"
    print(f"start time:{start_time}, start tick:{get_cn_stock_data_tick(start_time)}")
    print(f"default_start_timestamp:{get_stock_start_timestamp()}")
    print(f"seconds_per_unit:{unit_seconds('d')}")
