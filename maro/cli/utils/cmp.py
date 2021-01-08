from maro.cli.grass.agents.resource import NodeResource, ContainerResource


def resource_op(node_resource: dict, container_resource: dict, op: str):
    node_resource = NodeResource(
        node_name=node_resource["name"],
        cpu=node_resource["cpu"],
        memory=node_resource["memory"],
        gpu=node_resource["gpu"]
    )

    container_resource = ContainerResource(
        node_name=container_resource["name"],
        cpu=container_resource["cpu"],
        memory=container_resource["memory"],
        gpu=container_resource["gpu"]
    )
    is_satisfied = True

    if op == "allocate":
        if node_resource < container_resource:
            is_satisfied = False
        node_resource -= container_resource
    elif op == "release":
        node_resource += container_resource
    else:
        pass

    return is_satisfied, node_resource()
