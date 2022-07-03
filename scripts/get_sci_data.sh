#!/bin/bash

# The url of data blob.
blob_url="https://marodatasource.blob.core.windows.net/sci-topologies"

# The local dir of supply chain topologies.
topologies_dir="maro/simulator/scenarios/supply_chain/topologies"

# Download the data for topology plant & super_vendor.
wget ${blob_url}/plant/sample_preprocessed.csv -P ${topologies_dir}/plant

# Download the topology (config and data) of SCI_10.
wget https://marodatasource.blob.core.windows.net/sci-topologies/SCI_10.zip
unzip SCI_10.zip -d ${topologies_dir}
rm SCI_10.zip

# Download the topology (config and data) of SCI_500.
wget https://marodatasource.blob.core.windows.net/sci-topologies/SCI_500.zip
unzip SCI_500.zip -d ${topologies_dir}
rm SCI_500.zip
