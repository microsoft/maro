
import json
import numpy as np
from datetime import datetime



# modified version from: https://pypi.org/project/influx-line-protocol/
class Metric(object):

    def __init__(self, measurement):
        self.measurement = measurement
        self.values = {}
        self.tags = dict()
        self.timestamp = None

    def with_timestamp(self, timestamp=None):
        if timestamp is None:
            self.timestamp = datetime.timestamp(datetime.now())
        else:
            self.timestamp = timestamp

    def add_tag(self, name, value):
        self.tags[str(name)] = str(value)

    def add_value(self, name, value):
        self.values[str(name)] = value

    def __str__(self):
        # Escape measurement manually
        escaped_measurement = self.measurement.replace(',', '\\,')
        escaped_measurement = escaped_measurement.replace(' ', '\\ ')
        protocol = escaped_measurement

        # Create tag strings
        tags = []
        for key, value in self.tags.items():
            escaped_name = self.__escape(key)
            escaped_value = self.__escape(value)

            tags.append("%s=%s" % (escaped_name, escaped_value))

        # Concatenate tags to current line protocol
        if len(tags) > 0:
            protocol = "%s,%s" % (protocol, ",".join(tags))

        # Create field strings
        values = []
        for key, value in self.values.items():
            escaped_name = self.__escape(key)
            escaped_value = self.__parse_value(value)
            values.append("%s=%s" % (escaped_name, escaped_value))

        # Concatenate fields to current line protocol
        protocol = "%s %s" % (protocol, ",".join(values))

        if self.timestamp is not None:
            protocol = "%s %d" % (protocol, self.timestamp)

        return protocol

    def __escape(self, value, escape_quotes=False):
        # Escape backslashes first since the other characters are escaped with
        # backslashes
        new_value = value.replace('\\', '\\\\')
        new_value = new_value.replace(' ', '\\ ')
        new_value = new_value.replace('=', '\\=')
        new_value = new_value.replace(',', '\\,')

        if escape_quotes:
            new_value = new_value.replace('"', '\\"')

        return new_value

    def __parse_value(self, value):
        v_type = type(value)

        if v_type is int or v_type is np.int64 or v_type is np.int32 or v_type is np.int16:
            return "%di" % value

        if v_type is float or v_type is np.float:
            return "%g" % value

        if v_type is bool:
            return value and "t" or "f"

        if v_type is list or v_type is dict:
            value = json.dumps(value)

        # if type(value) is bytes or type(value) is bytearray:
        #     pass

        return "\"%s\"" % self.__escape(value, True)
