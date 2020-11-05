# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import calendar
import warnings
from csv import DictReader

from dateutil.parser import parse as parse_dt
from dateutil.tz import UTC, gettz

from maro.data_lib.common import SINGLE_BIN_FILE_TYPE, VERSION, dtype_convert_map, header_struct
from maro.data_lib.item_meta import BinaryMeta


def is_datetime(val: str, tzone=None):
    """Check if a string is datetime, return the datetime if yes."""
    try:
        # default time zone
        if tzone is None:
            tzone = UTC
        else:
            tzone = gettz(tzone)

        dt = parse_dt(val)

        dt = dt.replace(tzinfo=tzone)

        return True, dt
    except Exception:
        pass

    return False, None


def convert_val(val: str, dtype: str, tzone):
    """A simple function to convert str value into specified data type."""
    result = None

    # process the value first
    # clear
    val = val.strip("\"\'")
    # clear space at 2 side
    val = val.strip()

    # NOTE: we only support numeric value for now
    t = dtype_convert_map[dtype]

    try:
        # convert to float first, to avoid int("1.111") error
        v = float(val)

        result = t(v)
    except ValueError:
        # is it a datetime?
        is_dt, dt = is_datetime(val, tzone)

        if is_dt:
            # convert into UTC, then utc timestamp
            dt = dt.astimezone(UTC)
            result = calendar.timegm(dt.timetuple())

    if result is None:
        warnings.warn(f"Cannot parse value '{val}' into type '{dtype}'")

    return result


class BinaryConverter:
    """Convert csv file into binary with specified meta.

    The output binary file composed with:

    1. header: file type, start/end time etc.
    2. meta: meta content after validation.
    3. items.

    Args:
        output_file(str): Output binary file full path.
        meta_file(str): Path to the meta file (yaml).
        utc_start_timestamp(int): Start timestamp in UTC which will be considered as tick 0,
            used to adjust the data reader pipeline.

    """
    def __init__(self, output_file: str, meta_file: str, utc_start_timestamp: int = None):
        self._output_fp = None
        self._meta = BinaryMeta()
        self._meta.from_file(meta_file)

        self._output_fp = open(output_file, "wb+")

        self._item_count = 0
        self._item_size = self._meta.item_size
        self._meta_offset = header_struct.size
        self._meta_size = 0
        self._data_offset = 0
        self._data_size = 0
        self._starttime = 0
        self._endtime = 0

        # is starttime changed for 1st time
        self._is_starttime_changed = False

        # if we have a start timestamp, then use it in binary
        if utc_start_timestamp is not None:
            self._starttime = utc_start_timestamp

            # set it to True so that following logic will not change start time again
            self._is_starttime_changed = True

        # write header for 1st time, and meta
        self._update_header()
        self._write_meta()

    def add_csv(self, csv_file: str):
        """Convert specified csv file into current binary file, this converter will not sort the item.
        This method can be called several times to convert multiple csv file into one binary,
        the order will be same as calling sequence.

        Args:
            csv_file(str): Csv to convert.
        """
        with open(csv_file, newline='') as csv_fp:
            reader = DictReader(csv_fp)

            # write items
            self._write_items(reader)

    def flush(self):
        """Flush the result into output file."""
        self._update_header()

    def __del__(self):
        # resource collecting
        if self._output_fp is not None and not self._output_fp.closed:
            self.flush()

            self._output_fp.flush()
            self._output_fp.close()

    def _update_header(self):
        """Update file header."""
        header_bytes = header_struct.pack(
            b"MARO",
            SINGLE_BIN_FILE_TYPE,
            VERSION,
            self._item_count,
            self._item_size,
            self._meta_offset,
            self._meta_size,
            self._data_offset,
            self._data_size,
            self._starttime,
            self._endtime
        )

        self._meta_offset = len(header_bytes)
        # seek the output file beginning
        self._output_fp.seek(0, 0)
        self._output_fp.write(header_bytes)
        # seek to the file end
        self._output_fp.seek(0, 2)

    def _write_meta(self):
        """Write file meta."""
        meta_bytes = self._meta.to_bytes()

        # update header info
        self._data_offset = self._meta_offset + len(meta_bytes)
        self._meta_size = len(meta_bytes)

        self._output_fp.write(meta_bytes)

    def _write_items(self, reader: DictReader):
        """Write items into binary."""
        # columns need to convert
        columns = self._meta.columns
        # values buffer from each row, used to pack into binary
        values = [0] * len(columns.keys())
        # item binary buffer
        buffer = memoryview(bytearray(self._meta.item_size))
        # field -> data type
        field_type_dict = self._meta.items()
        # some column's value may cannot be parse, will skip it
        has_invalid_column = False

        for row in reader:
            field_index = 0
            has_invalid_column = False

            # clear the values
            for j in range(len(values)):
                values[j] = 0

            # read from current row
            for field, dtype in field_type_dict.items():
                column_name = columns[field]

                # NOTE: we allow field not exist in csv file, the value will be zero
                if column_name in row:
                    val = convert_val(row[column_name], dtype, self._meta.time_zone)
                    values[field_index] = val

                    if val is None:
                        has_invalid_column = True
                        break

                    # keep the start and end tick
                    if field == "timestamp":
                        if not self._is_starttime_changed:
                            self._is_starttime_changed = True
                            self._starttime = val
                        else:
                            self._starttime = min(self._starttime, val)

                        self._endtime = max(val, self._endtime)

                field_index += 1

            if not has_invalid_column:
                # convert item into bytes buffer, and write to file
                self._meta.item_to_bytes(values, buffer)
                self._output_fp.write(buffer)

                # update header fields for final update
                self._item_count += 1
                self._data_size += self._item_size


__all__ = ['BinaryConverter']
