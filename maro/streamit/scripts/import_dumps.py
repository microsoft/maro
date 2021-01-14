
from streamit.client import get_experiment_data_stream
import yaml
import numpy as np
import json
from argparse import ArgumentParser
import os


os.environ["MARO_STREAMABLE_ENABLED"] = "True"


def import_from_snapshot_dump(streamit, folder, npy_name, meta_name, category):
    npy_path = os.path.join(folder, npy_name)
    meta_path = os.path.join(folder, meta_name)

    field_names = None
    field_length = 0

    with open(meta_path, "r") as fp:
        field_name_list = fp.readline().split(",")
        field_length_list = [int(l) for l in fp.readline().split(",")]

    instance_list = np.load(npy_path)

    # instance number will be same for numpy backend
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

                #print("tick:", tick, "port index", port_index)

                if field_length == 1:
                    field_dict[field_name] = instance_list[tick][instance_index][field_name].item(
                    )
                else:
                    field_dict[field_name] = list(
                        [v.item() for v in instance_list[tick][instance_index][field_name]])

                field_slot_index += field_length

            streamit.data(category, **field_dict)

    instance_list = None

    return instance_number


def import_port_details(streamit, folder):
    port_npy_name = "ports.npy"
    port_meta_name = "ports.meta"
    category = "port_details"

    return import_from_snapshot_dump(streamit, folder, port_npy_name, port_meta_name, category)


def import_vessel_details(streamit, folder):
    vessels_npy_name = "vessels.npy"
    vessels_meta_name = "vessels.meta"
    category = "vessel_details"

    return import_from_snapshot_dump(streamit, folder, vessels_npy_name, vessels_meta_name, category)


def import_full_on_ports(data: np.ndarray, streamit, port_number):
    for tick in range(len(data)):
        streamit.tick(tick)

        m = data[tick][0].reshape(port_number, -1)

        # we only save cells that value > 0
        a, b = np.where(m > 0)

        for from_port_index, to_port_index in list(zip(a, b)):
            streamit.data("full_on_ports", from_port_index=from_port_index, dest_port_index=to_port_index, quantity=m[from_port_index, to_port_index])

def import_full_on_vessels(data: np.ndarray, streamit, port_number, vessel_number):
    for tick in range(len(data)):
        streamit.tick(tick)

        m = data[tick][0].reshape(vessel_number, port_number)

        a, b = np.where(m > 0)

        for vessel_index, port_index in list(zip(a, b)):
            streamit.data("full_on_vessels", vessel_index=vessel_index, port_index=port_index, quantity=m[vessel_index, port_index])

def import_vessel_plans(data, streamit, port_number, vessel_number):
    for tick in range(len(data)):
        streamit.tick(tick)

        m = data[tick][0].reshape(vessel_number, port_number)

        a, b = np.where(m > -1)

        for vessel_index, port_index in list(zip(a, b)):
            streamit.data("vessel_plans", vessel_index=vessel_index, port_index=port_index, planed_arrival_tick=m[vessel_index, port_index])


def import_metrics(streamit, epoch_full_path, port_number, vessel_number):
    matrics_path = os.path.join(epoch_full_path, "matrices.npy")

    matrics = np.load(matrics_path)

    import_full_on_ports(matrics["full_on_ports"],streamit, port_number)
    import_full_on_vessels(matrics["full_on_vessels"], streamit, port_number, vessel_number)
    import_vessel_plans(matrics["vessel_plans"], streamit, port_number, vessel_number)

    matrics = None


def import_attention():
    pass


if __name__ == "__main__":
    """"""
    parser = ArgumentParser()

    parser.add_argument("--name", required=True,
                        help="Experiment name show in databas")
    parser.add_argument("--scenario", required=True,
                        help="Scenario name of import experiment")
    parser.add_argument("--topology", required=True,
                        help="Topology of target scenario")
    parser.add_argument("--durations", required=True,
                        type=int, help="Durations of each episode")
    parser.add_argument("--episodes", required=True, type=int,
                        help="Total episode of this experiment")

    parser.add_argument("--dir", required=True,
                        help="Root folder of dump files")
    parser.add_argument(
        "--ssdir", help="Folder that contains snapshots data that with epoch_x sub-folders")

    parser.add_argument("--host", default="127.0.0.1",
                        help="Host of questdb server")

    args = parser.parse_args()

    assert(os.path.exists(args.dir))
    assert(os.path.exists(args.ssdir))

    streamit = get_experiment_data_stream(args.name)
    streamit.start(args.host)

    try:
        config = ""

        # experiment name
        with open(os.path.join(args.dir, "config.yml"), "r") as fp:
            config = yaml.safe_load(fp)

        streamit.info(args.scenario, args.topology,
                      args.durations, args.episodes)
        streamit.dict("config", config)

        for episode in range(args.episodes):
            epoch_folder = f"epoch_{episode}"

            epoch_full_path = os.path.join(args.ssdir, epoch_folder)

            # ensure epoch folder exist
            if os.path.exists(epoch_full_path):
                streamit.episode(episode)

                # import for each category
                port_number = import_port_details(streamit , epoch_full_path)

                vessel_number = import_vessel_details(
                    streamit, epoch_full_path)

                import_metrics(streamit, epoch_full_path,
                               port_number, vessel_number)

                # import_attention(streamit, episode, epoch_full_path)
    finally:
        streamit.close()
