
from collections import namedtuple

# actions for units


ConsumerAction = namedtuple("ConsumerAction", ("id", "source_id", "quantity", "vlt"))

ManufactureAction = namedtuple("ManufactureAction", ("id", "production_rate"))
