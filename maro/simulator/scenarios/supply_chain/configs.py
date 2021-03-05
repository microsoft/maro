

from .data import (
    StorageDataModel,
    TransportDataModel,
    DistributionDataModel,
    ManufactureDataModel,
    ConsumerDataModel
)

from .units import (
    StorageUnit,
    TransportUnit,
    DistributionUnit,
    ManufacturingUnit,
    ConsumerUnit
)

from .facilities import (
    WarehouseFacility,
    SupplierFacility
)


data_class_mapping = {
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
    },
    "ManufactureDataModel": {
        "alias_in_snapshot": "manufactures",
        "class": ManufactureDataModel
    },
    "ConsumerDataModel": {
        "alias_in_snapshot": "consumers",
        "class": ConsumerDataModel
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
    },
    "ManufacturingUnit": {
        "class": ManufacturingUnit
    },
    "ConsumerUnit": {
        "class": ConsumerUnit
    },
}

test_world_config = {
    # skus in this world, used to generate id
    "skus": [
        {
            "id": 1,
            "name": "sku1",
            "output_units_per_lot": 1,  # later we can support override per facility
            "bom": {    # bill of materials to procedure this, we can support facility level override
                "sku3": 10 # units per lot
            }
        },
        {
            "id": 2,
            "output_units_per_lot": 1,
            "name": "sku2"
        },
        {
            "id": 3,
            "output_units_per_lot": 1,
            "name": "sku3"
        }
    ],
    "facilities": [
        {
            # a source material supplier without input requirement
            "name": "Supplier3",
            "class": SupplierFacility,
            "configs": {
                "skus": {
                    "sku3": {
                        "init_in_stock": 100,
                        "production_rate": 200,
                        "type": "production",
                        "cost": 10,
                        "price": 10
                    }
                },
                "storage": {
                    "data": {
                        "capacity": 20000,
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
        },
        {
            "name": "Supplier1",
            "class": SupplierFacility,
            "configs": {
                "skus": {
                    "sku1": {
                        "init_in_stock": 100,
                        "production_rate": 200,
                        "type": "production",
                        "cost": 10,
                        "price": 100
                    },
                    # source material, do not need production rate
                    "sku3": {
                        "init_in_stock": 100,
                        "production_rate": 200,
                        "type": "material",
                        "cost": 10,
                        "price": 100
                    }
                },
                "storage": {
                    "data": {
                        "capacity": 20000,
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
        },
        {
            "name": "warehouse1",
            "class": WarehouseFacility,
            "configs": {
                "skus": {
                    "sku1": {
                        "init_stock": 1000,
                        "price": 100
                    },
                    "sku2": {
                        "init_stock": 1000,
                        "price": 100
                    },
                    "sku3": {
                        "init_stock": 1000,
                        "price": 100
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
    ],
    # topology used to specify the up/downstream for facilities
    # we split it from facility, so that we can support configuration inherit to override it
    # for a new topology
    "topology": {
        # key is current facility, value if upstream facilities that will provide a certain sku
        "Supplier1": {
            # this config means "Supplier1" will purchase "sku3" from facility "Supplier3",
            # or any other facility in the list
            "sku3": [
                "Supplier3"
            ]
        },
        "warehouse1": {
            "sku1": [
                "Supplier1"
            ],
            "sku3": [
                "Supplier3"
            ]
        }
    }
}
