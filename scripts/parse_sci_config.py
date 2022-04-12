import collections
import os
import sys
from typing import Generator

import pandas as pd
import pyarrow.parquet as pq
import yaml

from tqdm import tqdm

class Parser:
    def __init__(self, workspace: str) -> None:
        self._workspace = workspace

        self._skus = []
        self._bom = collections.defaultdict(list)
        self._sku_config = []

        self._facility_config = []
        self._world_config = {}

        self._topology = {}

    def readlines(self, path: str) -> Generator[pd.Series, None, None]:
        path = os.path.join(self._workspace, "csv", path)

        df = pq.read_table(path).to_pandas() if path.endswith('.parquet') else pd.read_csv(path)
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=path):
            yield row

    def _parse_skus(self) -> None:
        self._skus = sorted([row['ItemId'] for row in self.readlines("item.csv")])

        for row in self.readlines("billofmateriallines.csv"):
            item_id, component_id, qty = row["ItemId"], row["ComponentItemId"], row["ComponentQuantity"]
            self._bom[item_id].append((component_id, qty))

        for i, sku_name in enumerate(self._skus):
            cur_sku_config = {
                "id": i + 1,
                "name": sku_name,
                "output_units_per_lot": 1,  # TODO
            }
            if sku_name in self._bom:
                cur_sku_config["bom"] = {material_id: qty for material_id, qty in self._bom[sku_name]}
            self._sku_config.append(cur_sku_config)

    def _parse_facilities(self) -> None:
        # Facility
        sku_assignment = collections.defaultdict(list)
        for row in self.readlines("production_assignment.csv"):
            sku, supplier_name = row["ProductId"], row["ProductionPlantId"]
            sku_assignment[supplier_name].append(sku)

        for row in self.readlines("production_plant.csv"):
            name = row["ProductionPlantId"]
            self._facility_config.append({
                "name": f"Supplier__{name}",
                "definition_ref": "SupplierFacility",
                "skus": {
                    sku: {  # TODO
                        "init_stock": 100,
                        "product_unit_cost": 1,
                        "production_rate": 1,
                        "type": "production",
                        "cost": 10,
                        "price": 10,
                        "vlt": 1,
                    }
                    for sku in sorted(sku_assignment[name])
                },
            })

        # Warehouse & Supplier
        warehouses = []
        retailers = []
        for row in self.readlines("warehouse.csv"):
            name = row["WarehouseId"]
            if name.startswith("PLANT_"):
                continue
            if name.startswith("STRG_"):
                warehouses.append({  # TODO: sku details?
                    "name": f"Warehouse__{name}",
                    "definition_ref": "WarehouseFacility",
                })
            else:
                retailers.append({
                    "name": f"Retailer__{name}",
                    "definition_ref": "RetailerFacility",
                })
        self._facility_config.extend(warehouses)
        self._facility_config.extend(retailers)

    def _parse_topology(self) -> None:
        warehouse_items = collections.defaultdict(set)
        for row in self.readlines("orderline.csv"):
            warehouse_id, item_id = row["WarehouseId"], row["ItemId"]
            warehouse_items[warehouse_id].add(item_id)

        for warehouse_id, items in warehouse_items.items():
            city = warehouse_id.split("_")[0]
            self._topology[warehouse_id] = {
                item_id: [facility["name"] for facility in self._facility_config if f"STRG_{city}" in facility["name"]]
                for item_id in sorted(list(items))
            }

    def _parse_world(self) -> None:
        self._parse_facilities()
        self._parse_topology()

        self._world_config = {
            "skus": self._sku_config,
            "facilities": self._facility_config,
            "topology": self._topology,
        }

    def parse(self) -> None:
        self._parse_skus()
        self._parse_world()

        total_config = {
            "skus": self._sku_config,
            "world": self._world_config,
        }
        with open(os.path.join(self._workspace, "config.yml"), "w") as fp:
            yaml.safe_dump(total_config, fp)


if __name__ == '__main__':
    parser = Parser(sys.argv[1])
    parser.parse()
