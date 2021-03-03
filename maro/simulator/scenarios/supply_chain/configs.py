

from .datamodels import (
    StorageDataModel,
    TransportDataModel,
    DistributionDataModel
)

from .logics import (
    StorageLogic,
    SimpleTransportLogic,
    DistributionLogic
)

from .units import (
    StorageUnit,
    TransportUnit,
    DistributionUnit
)

from .facilities import (
    WarehouseFacility
)


datamodel_mapping = {
    "StorageDataModel": {
        "alias_in_snapshot": "storages",
        "class": StorageDataModel
    },
    "TransportDataModel": {
        "alias_in_snapshot": "transports",
        "class": TransportDataModel
    },
    "DistributionDataModel": {
        "alias_in_snapshot": "distributions",
        "class": DistributionDataModel
    }
}


logic_mapping = {
    "StorageLogic": {
        "class": StorageLogic
    },
    "TransportLogic": {
        "class": SimpleTransportLogic
    },
    "DistributionLogic": {
        "class": DistributionLogic
    }
}

unit_class_mapping = {
    "StorageUnit": {
        "class": StorageUnit
    },
    "TransportUnit": {
        "class": TransportUnit
    }
}


test_world_config = {
    "facilities": {
        "warehouse1": {
            "class": WarehouseFacility,
            "configs": {
                "skus": {
                    "sku1": {
                        "price": 100,
                        "cost": 100,
                        "vlt": 5,
                        "init_stock": 1000,
                        "production_rate": 200
                    },
                    "sku2": {
                        "price": 100,
                        "cost": 100,
                        "vlt": 5,
                        "init_stock": 1000,
                        "production_rate": 200
                    },
                    "sku3": {
                        "price": 100,
                        "cost": 100,
                        "vlt": 5,
                        "init_stock": 1000,
                        "production_rate": 200
                    },
                },
                "storage": {
                    "data": {
                        "capacity": 200,
                        "unit_storage_cost": 10
                    }
                },
                "distribution": {
                    "data": {
                        "unit_price": 10
                    }
                },
                "transports": [
                    {
                        "data": {
                            "patient": 100
                        }
                    },
                    {
                        "data": {
                            "patient": 100
                        }
                    }
                ]
            }
        }
    }
}
