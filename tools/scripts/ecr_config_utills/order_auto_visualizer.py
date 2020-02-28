# this script used to show a chart for specified order config
import sys
import getopt
import numpy as np
import matplotlib.pyplot as plt

from maro.simulator.scenarios.ecr.ecr_data_generator import EcrDataGenerator


TOPOLOGY_PATH = "../../../maro/simulator/scenarios/ecr/topologies/"


def draw_spacial_distribution(config_yml_path, max_tick, save_fig=False):
    data_generator: EcrDataGenerator = EcrDataGenerator(max_tick, config_yml_path)

    port_names = list(data_generator._ports._conf.keys())
    sorted_port_names = sorted(port_names)
    indices = [sorted_port_names.index(name) for name in port_names]

    def compress(name: str):
        return name[-1] if name.isdigit() else name[0]

    abbreviations = [''.join([compress(word) for word in name.split('_')]) for name in sorted_port_names]

    order_qty_list = []
    order_matrix = np.zeros((data_generator.port_num, data_generator.port_num))
    for tick in range(max_tick):
        orders = data_generator.generate_orders(tick, data_generator._container_proportion._total_container)
        order_qty_counter = 0
        for order in orders:
            order_qty = order.quantity
            order_qty_counter += order_qty
            order_matrix[indices[order.src_port_idx]][indices[order.dest_port_idx]] += order_qty
        order_qty_list.append(order_qty_counter)

    fig, (ax0, ax1) = plt.subplots(2, 1)
    ax0.pcolor(order_matrix, cmap='OrRd')
    ax0.set_xticks([i + 0.5 for i in range(len(sorted_port_names))])
    ax0.set_yticks([i + 0.5 for i in range(len(sorted_port_names))])
    if "22p_global" not in config_yml_path:
        for i in range(data_generator.port_num):
            for j in range(data_generator.port_num):
                ax0.text(i + 0.5, j + 0.5, order_matrix[j, i], horizontalalignment='center', verticalalignment='center')
        ax0.set_xticklabels(abbreviations)
        ax0.set_yticklabels(sorted_port_names)
    else:
        ax0.set_xticklabels(abbreviations, fontsize=6)
        ax0.set_yticklabels(sorted_port_names, fontsize=6)
    ax0.set_title('Order Matrix')
    plt.plot(np.arange(len(order_qty_list)), order_qty_list, color='tab:red')
    ax1.set(xlabel='tick', ylabel='order qty',
            title='Usage Proportion Distribution')
    fig.tight_layout()
    if save_fig:
        plt.savefig(config_yml_path[:-10] + "order_summary.png")
        plt.close()
    else:
        plt.show()


def draw_all_orders(max_ticks):
    for topology in ["4p_ssdd", "5p_ssddd", "6p_sssbdd", "22p_global_trade"]:
        for level in range(9):
            config_yml_path = TOPOLOGY_PATH + topology + "_l0." + str(level) + "/config.yml"
            print("Drawing", topology, "level", level, "...")
            draw_spacial_distribution(config_yml_path, max_ticks, save_fig=True)


if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "ht:m:c:", ["ticks=", "mode=", "config="])
    except getopt.GetoptError:
        sys.exit(1)

    draw_all_orders(224)

