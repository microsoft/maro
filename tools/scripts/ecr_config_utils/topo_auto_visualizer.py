import glob
import yaml
import copy

import matplotlib.pyplot as plt
import networkx as nx
import palettable.cmocean.sequential as cb


TOPOLOGY_PATH = "../maro/simulator/scenarios/ecr/topologies/"


def topo_visualizer_fixpos(nodes, node_groups, edges, edge_groups, path, route_groups, route=True, order=True):
    # G = nx.Graph()
    G = nx.MultiDiGraph()

    # node property
    for node in nodes:
        G.add_node(node[0], weight=node[1])

    # edge property
    reverse = []
    for edge in edges:
        if G.number_of_edges(edge[1], edge[0]) < 1:
            G.add_edge(edge[0], edge[1], weight=edge[2])
        else:
            G.add_edge(edge[0], edge[1], weight=edge[2])
            reverse.append(edge)

    for edge in reverse:
        G.remove_edge(edge[0], edge[1])

    pos = None
    if '4p' in path:
        pos = {'supply_port_002': [-0.9, 0.5], 'supply_port_001': [-0.9, -0.5],
               'demand_port_001': [0.9, -0.5], 'demand_port_002': [0.9, 0.5]}
    elif '5p' in path:
        pos = {'transfer_port_001': [-0., -0.], 'supply_port_001': [-0.9, 0.5], 'supply_port_002': [-0.9, -0.5],
               'demand_port_001': [0.9, -0.5], 'demand_port_002': [0.9, 0.5]}
    elif '6p' in path:
        pos = {'transfer_port_001': [0.4, -0.], 'transfer_port_002': [-0.4, -0.], 'supply_port_001': [-0.9, 0.5],
               'supply_port_002': [-0.9, -0.5],
               'demand_port_001': [0.9, -0.5], 'demand_port_002': [0.9, 0.5]}
    else:
        assert "Please use visualizer without fixed pos!"

    # color_bar = ["b","g","r","c","m","y","k"]
    node_labels = dict((n, d['weight']) for n, d in G.nodes(data=True))
    edge_labels = dict(((n, v), round(d['weight'], 2)) for n, v, d in G.edges(data=True))
    num_color = len(node_groups) if len(node_groups) > 2 else 3
    colormap = 'cb.Matter_' + str(num_color)
    colors = list(eval(colormap).hex_colors)

    if order:
        # draw node
        for i, node_group in enumerate(node_groups):
            nx.draw_networkx_nodes(G, pos, node_color=colors[i], alpha=1, nodelist=node_group, node_size=600)

        for p in pos:
            pos[p][0] -= 0.05  # 0.01, 0.04
        # draw edge and its label
        for i, edge_group in enumerate(edge_groups):
            single_edge_group = [(u, v) for (u, v, d) in G.edges(data=True) if (u, v) in edge_group]
            nx.draw_networkx_edges(G, pos, edgelist=single_edge_group, width=5, alpha=0.8, edge_color=colors[i])
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, alpha=0.8, rotate=True)

        # draw node labels
        for p in pos:  # raise text positions
            pos[p][0] += 0.05  # 0.01, 0.04
            pos[p][1] += 0.08  # 0.06, 0.08
        nx.draw_networkx_labels(G, pos, font_size=9, font_family='sans-serif')
        for p in pos:
            pos[p][1] -= 0.16  # 0.12, 0.16
        nx.draw_networkx_labels(G, pos, font_size=9, labels=node_labels, font_family='sans-serif')

        # draw reversed edge and its label
        for p in pos:  # raise text positions
            # pos[p][1] += 0.05 # 0.07, 0.10
            pos[p][0] += 0.05  # 0.01, 0.04
        G.clear()
        for edge in reverse:
            G.add_edge(edge[0], edge[1], weight=edge[2])
        reverse_edge_labels = dict(((n, v), round(d['weight'], 2)) for n, v, d in G.edges(data=True))
        for i, edge_group in enumerate(edge_groups):
            single_edge_group = [(u, v) for (u, v, d) in G.edges(data=True) if (u, v) in edge_group]
            nx.draw_networkx_edges(G, pos, edgelist=single_edge_group, width=5, alpha=0.8, edge_color=colors[i])
        nx.draw_networkx_edge_labels(G, pos, edge_labels=reverse_edge_labels, font_size=9, alpha=0.8, rotate=True)

    # draw route graph
    if route:
        G.clear()
        y_min = 10
        for p in pos:  # raise text positions
            pos[p][1] += 0.08
            pos[p][0] += 2.5
            y_min = min(pos[p][1], y_min)
        for route in route_groups:
            for edge in route:
                G.add_edge(edge[0], edge[1])
        for i, node_group in enumerate(node_groups):
            nx.draw_networkx_nodes(G, pos, node_color=colors[i], alpha=1, nodelist=node_group, node_size=600)
        for i, route_group in enumerate(route_groups):
            # single_edge_group = [(u, v) for (u, v, d) in G.edges(data=True) if (u,v) in edge_group]
            nx.draw_networkx_edges(G, pos, edgelist=route_group, width=5, alpha=0.8, edge_color=colors[i])
        for p in pos:  # raise text positions
            pos[p][1] += 0.08
        nx.draw_networkx_labels(G, pos, font_size=9, font_family='sans-serif')

    G.clear()
    # G.add_edge("Figure1.Order distribution topology","Figure2.Route distribution topology")
    # pos = {"Figure1.Order distribution topology":(-0.0,y_min),"Figure2.Route distribution topology":(2.5,y_min)}
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

    plt.axis('off')
    typ = 'route' if route else 'order_distribution'
    plt.savefig(path + '/' + typ + '.eps', format='eps')
    plt.show()
    plt.cla()

    return None


def topo_visualizer(nodes, node_groups, edges, edge_groups, path, route_groups, route=True, order=True):
    # G = nx.Graph()
    G = nx.MultiDiGraph()

    # node property
    for node in nodes:
        G.add_node(node[0], weight=node[1])

    # edge property
    reverse = []
    for edge in edges:
        if G.number_of_edges(edge[1], edge[0]) < 1:
            G.add_edge(edge[0], edge[1], weight=edge[2])
        else:
            G.add_edge(edge[0], edge[1], weight=edge[2])
            reverse.append(edge)

    # positions for all nodes
    pos = nx.spring_layout(G)
    pos_2 = copy.deepcopy(pos)
    for edge in reverse:
        G.remove_edge(edge[0], edge[1])

    # color_bar = ["b", "g", "r", "c", "m", "y", "k"]
    node_labels = dict((n, d['weight']) for n, d in G.nodes(data=True))
    edge_labels = dict(((n, v), round(d['weight'], 2)) for n, v, d in G.edges(data=True))
    num_color = len(node_groups) if len(node_groups) > 2 else 3
    colormap = 'cb.Matter_' + str(num_color)
    colors = list(eval(colormap).hex_colors)

    if order:
        # draw node
        for i, node_group in enumerate(node_groups):
            nx.draw_networkx_nodes(G, pos, node_color=colors[i], alpha=1, nodelist=node_group, node_size=400)

        # draw edge and its label
        for i, edge_group in enumerate(edge_groups):
            single_edge_group = [(u, v) for (u, v, d) in G.edges(data=True) if (u, v) in edge_group]
            nx.draw_networkx_edges(G, pos, edgelist=single_edge_group, width=3, alpha=0.5, edge_color=colors[i])
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=5, alpha=0.8, rotate=True)

        # draw node labels
        for p in pos:  # raise text positions
            pos[p][1] += 0.08  # 0.06, 0.08
        nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')
        for p in pos:
            pos[p][1] -= 0.16  # 0.12, 0.16
        nx.draw_networkx_labels(G, pos, font_size=8, labels=node_labels, font_family='sans-serif')

        # draw reversed edge and its label
        for p in pos:  # raise text positions
            pos[p][1] += 0.02  # 0.07, 0.10
            pos[p][0] += 0.02  # 0.01, 0.04
        G.clear()
        for edge in reverse:
            G.add_edge(edge[0], edge[1], weight=edge[2])
        reverse_edge_labels = dict(((n, v), round(d['weight'], 2)) for n, v, d in G.edges(data=True))
        for i, edge_group in enumerate(edge_groups):
            single_edge_group = [(u, v) for (u, v, d) in G.edges(data=True) if (u, v) in edge_group]
            nx.draw_networkx_edges(G, pos, edgelist=single_edge_group, width=3, alpha=0.5, edge_color=colors[i])
        # nx.draw_networkx_edge_labels(G, pos, edge_labels = reverse_edge_labels, font_size=5, alpha=0.8, rotate=True)

    if route:
        # draw route graph
        G.clear()
        y_min = 10
        for p in pos:
            pos[p][1] += 3
            y_min = min(pos[p][1], y_min)
        for route in route_groups:
            for edge in route:
                G.add_edge(edge[0], edge[1])
        for i, node_group in enumerate(node_groups):
            nx.draw_networkx_nodes(G, pos, node_color=colors[i], alpha=1, nodelist=node_group, node_size=400)
        for i, route_group in enumerate(route_groups):
            # single_edge_group = [(u, v) for (u, v, d) in G.edges(data=True) if (u,v) in edge_group]
            nx.draw_networkx_edges(G, pos, edgelist=route_group, width=3, alpha=0.5, edge_color=colors[i])
        for p in pos:  # raise text positions
            pos[p][1] += 0.06
        nx.draw_networkx_labels(G, pos, font_size=8, font_family='sans-serif')

    plt.rcParams['figure.figsize'] = (200, 100)
    plt.axis('off')
    typ = 'route' if route else 'order_distribution'
    plt.savefig(path + '/' + typ + '.eps', format='eps')
    plt.show()
    plt.cla()

    return None


def parse_topo(source_path):
    with open(source_path, "r") as yaml_file:
        # Parse original business engine in config yml
        source_yaml_obj = yaml.load(yaml_file.read(), Loader=yaml.Loader)
        ports = source_yaml_obj["ports"]
        routes = source_yaml_obj["routes"]

    # port
    nodes = []
    edges = []
    for port in ports:
        if "source" in ports[port]["order_distribution"]:
            node = [port, ports[port]["order_distribution"]['source']['proportion']]
            nodes.append(node)

        if "targets" in ports[port]["order_distribution"]:
            targets = ports[port]["order_distribution"]["targets"]
            for target in targets:
                edge = [port, target, targets[target]['proportion']]
                edges.append(edge)

    # route
    edge_groups = []
    route_groups = []
    node_groups = []
    for route in routes:
        edge_group = []
        route_group = []
        node_group = []
        for i, port in enumerate(routes[route][:-1]):
            node_group.append(port["port_name"])
            route_group.append((port["port_name"], routes[route][i + 1]["port_name"]))
        node_group.append(routes[route][-1]["port_name"])
        route_group.append((routes[route][-1]["port_name"], routes[route][0]["port_name"]))
        for i, port in enumerate(routes[route]):
            for j in range(i + 1, len(routes[route])):
                edge_group.append((port["port_name"], routes[route][j]["port_name"]))
                edge_group.append((routes[route][j]["port_name"], port["port_name"]))

        node_groups.append(node_group)
        edge_groups.append(edge_group)
        route_groups.append(route_group)

    return nodes, node_groups, edges, edge_groups, route_groups


if __name__ == '__main__':
    pathlist = glob.glob(TOPOLOGY_PATH + "*p*")
    route_flag = False
    order_flag = True
    for path in pathlist:
        nodes, node_groups, edges, edge_groups, route_groups = parse_topo(path + "/config.yml")
        if '22p' in path:
            topo_visualizer(nodes, node_groups, edges, edge_groups, path, route_groups, route=route_flag,
                            order=order_flag)
        else:
            topo_visualizer_fixpos(nodes, node_groups, edges, edge_groups, path, route_groups, route=route_flag,
                                   order=order_flag)
