1. distribution:
    move delay_order_penalty configuration from facility to sku
   
2. manufacturing:
    original: 
        . bom not configured in configuration file
        . source material is same as output product
        . hard coded "output_lot_size = 1" and control.production_rate to control the produce rate,
          but the sku.production_rate in configuration file not in use.
   
    changed:
        . bom configured in configuration file at world level. later we can support override this at each manufacture unit.
        . remove output_lot_size from bom, user sku.production_rate at facility level, then action can change this
        . support manufacturing without source material, like oil, just produce output production by configured rate
        . add type for sku at facility level to identify if it is an input material, or output production
        . remove output_lot_size, always be 1
   
3. consumer:
    split upstreams from facility to a standalone part, so that we can override same world structure but
   different topology.
   
    . there is no consumer_quantity and source by default, we use first source as source_id
        but with quantity as 0, means we will wait for action to purchase source sku.