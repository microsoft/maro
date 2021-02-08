# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import numpy as np

from datetime import datetime


def escape(value: str, escape_quotes=False):
    """Escape string value.

    Args:
        value (str): Value to escape.
        escape_quotes (bool): If we should escape quotes.

    return:
        str: Escaped string value.
    """
    # Escape backslashes first since the other characters are escaped with
    # backslashes
    new_value = value.replace('\\', '\\\\')
    new_value = new_value.replace(' ', '\\ ')
    new_value = new_value.replace('=', '\\=')
    new_value = new_value.replace(',', '\\,')

    if escape_quotes:
        new_value = new_value.replace('"', '\\"')

    return new_value


def parse_value(value: object):
    """"Parse value into string to fit influxdb line protocol.

    Args:
        value (object): Value to parse.

    Returns:
        str: String format of value.
    """
    v_type = type(value)

    if v_type is int or v_type is np.int64 or v_type is np.int32 or v_type is np.int16:
        return "%di" % value

    if v_type is float or v_type is np.float or v_type is np.float32 or v_type is np.float64:
        return "%g" % value

    if v_type is bool:
        return value and "t" or "f"

    if v_type is list or v_type is dict:
        value = json.dumps(value)

    if v_type is np.ndarray:
        value = json.dumps(value.tolist())

    return "\"%s\"" % escape(value, True)


# modified version from: https://pypi.org/project/influx-line-protocol/
class Metric(object):
    """Metric used to convert message into to influxdb line protocol message.

    Args:
        measurement (str): Name of the measurement of current message.
    """
    def __init__(self, measurement: str):
        self.measurement = measurement
        self.values = {}
        self.tags = {}
        self.timestamp = None

    def with_timestamp(self, timestamp=None):
        """Add timestamp into message.

        Args:
            timestamp (int): Timestamp to send, None to used current system timestamp. Default is None.
        """
        if timestamp is None:
            self.timestamp = datetime.timestamp(datetime.now())
        else:
            self.timestamp = timestamp

    def add_tag(self, name: str, value: str):
        """Add a tag to current message.

        Args:
            name (str): Tag name.
            value (str): Value of this tag.
        """
        self.tags[str(name)] = str(value)

    def add_value(self, name: str, value: object):
        """Add a named value.

        Args:
            name (str): Name of the value (column).
            value (object): Value to add.
        """
        self.values[str(name)] = value

    def __str__(self):
        # Escape measurement manually.
        escaped_measurement = self.measurement.replace(',', '\\,')
        escaped_measurement = escaped_measurement.replace(' ', '\\ ')
        protocol = escaped_measurement

        # Create tag strings.
        tags = []
        for key, value in self.tags.items():
            escaped_name = escape(key)
            escaped_value = escape(value)

            tags.append("%s=%s" % (escaped_name, escaped_value))

        # Concatenate tags to current line protocol.
        if len(tags) > 0:
            protocol = "%s,%s" % (protocol, ",".join(tags))

        # Create field strings.
        values = []
        for key, value in self.values.items():
            escaped_name = escape(key)
            escaped_value = parse_value(value)
            values.append("%s=%s" % (escaped_name, escaped_value))

        # Concatenate fields to current line protocol.
        protocol = "%s %s" % (protocol, ",".join(values))

        if self.timestamp is not None:
            protocol = "%s %d" % (protocol, self.timestamp)

        return protocol


__all__ = ['Metric']
