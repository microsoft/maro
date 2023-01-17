# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pickle
from argparse import ArgumentParser

import numpy as np
import yaml


def import_from_snapshot_dump(streamit, folder: str, npy_name: str, meta_name: str, category: str):
    """Import specified category from snapshot dump file into data service.

    Args:
        streamit (streamit) : Streamit instance.
        folder (str): Folder name of snapshot dump file.
        npy_name (str): Name of .npy file that hold dumped numpy array data.
        meta_name (str): File name of the meta file.
        category (str): Category name to save into database.
    """
    npy_path = os.path.join(folder, npy_name)
    meta_path = os.path.join(folder, meta_name)

    # Read meta file to get names and length of each field.
    with open(meta_path, "r") as fp:
        field_name_list = fp.readline().split(",")
        field_length_list = [int(line) for line in fp.readline().split(",")]

    instance_list: np.ndarray = np.load(npy_path)

    # Instance number will be same for numpy backend.
    instance_number = len(instance_list[0])

    for tick in range(len(instance_list)):
        streamit.tick(tick)

        for instance_index in range(instance_number):
            field_dict = {}

            field_slot_index = 0

            for field_index in range(len(field_name_list)):
                field_name = field_name_list[field_index].strip()
                field_length = field_length_list[field_index]

                field_dict["index"] = instance_index

                if field_length == 1:
                    field_dict[field_name] = instance_list[tick][instance_index][field_name].item()
                else:
                    field_dict[field_name] = list(
                        [v.item() for v in instance_list[tick][instance_index][field_name]],
                    )

                field_slot_index += field_length

            streamit.data(category, **field_dict)

    return instance_number


def import_port_details(streamit, folder: str):
    """Import port details into database from specified folder.

    Args:
        streamit (streamit) : Streamit instance.
        folder (str): Folder path that contains the port detail file.
    """
    port_npy_name = "ports.npy"
    port_meta_name = "ports.meta"
    category = "port_details"

    return import_from_snapshot_dump(streamit, folder, port_npy_name, port_meta_name, category)


def import_vessel_details(streamit, folder: str):
    """Import vessel details into database.

    Args:
        streamit (streamit) : Streamit instance.
        folder (str): Folder path that contains vessel details.
    """
    vessels_npy_name = "vessels.npy"
    vessels_meta_name = "vessels.meta"
    category = "vessel_details"

    return import_from_snapshot_dump(streamit, folder, vessels_npy_name, vessels_meta_name, category)


def import_full_on_ports(streamit, data: np.ndarray, port_number: int):
    """Import full_on_ports information into database.

    Args:
        streamit (streamit) : Streamit instance.
        data (numpy.ndarray): Data of full_on_ports.
        port_number (int): Number of ports.
    """
    for tick in range(len(data)):
        streamit.tick(tick)

        m = data[tick][0].reshape(port_number, -1)

        # We only save cells that value > 0.
        a, b = np.where(m > 0)

        for from_port_index, to_port_index in list(zip(a, b)):
            streamit.data(
                "full_on_ports",
                from_port_index=from_port_index,
                dest_port_index=to_port_index,
                quantity=m[from_port_index, to_port_index],
            )


def import_full_on_vessels(streamit, data: np.ndarray, port_number: int, vessel_number: int):
    """Import full_on_vessels data into database.

    Args:
        streamit (streamit) : Streamit instance.
        data (numpy.ndarray): Data that contains full_on_vessels matrix.
        port_number (int): Number of ports.
        vessel_number (int): Number of vessels.
    """
    for tick in range(len(data)):
        streamit.tick(tick)

        m = data[tick][0].reshape(vessel_number, port_number)

        a, b = np.where(m > 0)

        for vessel_index, port_index in list(zip(a, b)):
            streamit.data(
                "full_on_vessels",
                vessel_index=vessel_index,
                port_index=port_index,
                quantity=m[vessel_index, port_index],
            )


def import_vessel_plans(streamit, data: np.ndarray, port_number: int, vessel_number: int):
    """Import vessel_plans matrix into database.

    Args:
        streamit (streamit) : Streamit instance.
        data (numpy.ndarray): Data that contains vessel_plans matrix.
        port_number (int): Number of ports.
        vessel_number (int): Number of vessels.
    """
    for tick in range(len(data)):
        streamit.tick(tick)

        m = data[tick][0].reshape(vessel_number, port_number)

        a, b = np.where(m > -1)

        for vessel_index, port_index in list(zip(a, b)):
            streamit.data(
                "vessel_plans",
                vessel_index=vessel_index,
                port_index=port_index,
                planed_arrival_tick=m[vessel_index, port_index],
            )


def import_metrics(streamit, epoch_full_path: str, port_number: int, vessel_number: int):
    """Import matrix into database.

    Args:
        streamit (streamit) : Streamit instance.
        epoch_full_path (str): Path that for target epoch.
        port_number (int): Number of ports.
        vessel_number (int): Number of vessels.
    """
    matrics_path = os.path.join(epoch_full_path, "matrices.npy")

    matrics = np.load(matrics_path)

    import_full_on_ports(streamit, matrics["full_on_ports"], port_number)
    import_full_on_vessels(streamit, matrics["full_on_vessels"], port_number, vessel_number)
    import_vessel_plans(streamit, matrics["vessel_plans"], port_number, vessel_number)


def import_attention(streamit, atts_path: str):
    """Import attaention data.

    Args:
        streamit (streamit) : Streamit instance.
        atts_path (str): Path to attention file.
    """
    with open(atts_path, "rb") as fp:
        attentions = pickle.load(fp)

    attention_index = -1

    # List of tuple (tick, attention dict contains:"p2p", "p2v", "v2p").
    for tick, attention in attentions:
        attention_index += 1

        tick = int(tick)

        streamit.tick(tick)

        streamit.complex("attentions", attention)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--name",
        required=True,
        help="Experiment name show in database",
    )
    parser.add_argument(
        "--scenario",
        required=True,
        help="Scenario name of import experiment",
    )
    parser.add_argument(
        "--topology",
        required=True,
        help="Topology of target scenario",
    )
    parser.add_argument(
        "--durations",
        required=True,
        type=int,
        help="Durations of each episode",
    )
    parser.add_argument(
        "--episodes",
        required=True,
        type=int,
        help="Total episode of this experiment",
    )

    parser.add_argument(
        "--dir",
        required=True,
        help="Root folder of dump files",
    )
    parser.add_argument(
        "--ssdir",
        help="Folder that contains snapshots data that with epoch_x sub-folders",
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host of questdb server",
    )

    args = parser.parse_args()

    assert os.path.exists(args.dir)
    assert os.path.exists(args.ssdir)

    # Force enable streamit.
    os.environ["MARO_STREAMIT_ENABLED"] = "true"
    os.environ["MARO_STREAMIT_EXPERIMENT_NAME"] = args.name

    from maro.streamit import streamit

    with streamit:
        # experiment name
        with open(os.path.join(args.dir, "config.yml"), "r") as fp:
            config = yaml.safe_load(fp)

        # streamit.info(args.scenario, args.topology, args.durations, args.episodes)
        streamit.complex("config", config)

        for episode in range(args.episodes):
            epoch_folder = f"epoch_{episode}"

            epoch_full_path = os.path.join(args.ssdir, epoch_folder)

            # ensure epoch folder exist
            if os.path.exists(epoch_full_path):
                streamit.episode(episode)

                # import for each category
                port_number = import_port_details(streamit, epoch_full_path)

                vessel_number = import_vessel_details(streamit, epoch_full_path)

                import_metrics(streamit, epoch_full_path, port_number, vessel_number)

        # NOTE: we only have one attention file for now, so hard coded here
        streamit.episode(0)
        import_attention(streamit, os.path.join(args.dir, "atts_1"))
