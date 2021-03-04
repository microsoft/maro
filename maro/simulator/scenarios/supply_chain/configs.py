

from .datamodels import (
    StorageDataModel,
    TransportDataModel,
    DistributionDataModel
)

from .unit import (
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


unit_mapping = {
    "StorageUnit": {
        "class": StorageUnit
    },
    "TransportUnit": {
        "class": TransportUnit
    },
    "DistributionUnit": {
        "class": DistributionUnit
    }
}

test_world_config = {
    # skus in this world, used to generate id
    "skus": [
        {
            "id": 1,
            "name": "sku1"
        },
        {
            "id": 2,
            "name": "sku2"
        },
        {
            "id": 3,
            "name": "sku3"
        }
    ],
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
