import json
import os
import time
import traceback
from enum import Enum

import altair as alt
import pandas as pd
import streamlit as st

from maro.cli.grass.utils.params import JobStatus


class DashboardType(Enum):
    PROCESS = "process"
    LOCAL = "local"
    AZURE = "azure"
    ONPREMISES = "on-premises"


class DataType(Enum):
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"


def write_static_resurce(root, data, node_name, dashboard_type):
    if dashboard_type in [DashboardType.PROCESS, DashboardType.LOCAL]:
        root.header("Overall Resource:")
    else:
        root.header("Cluster Resource:")

    col1, col2, col3 = root.beta_columns(3)
    with col1:
        st.subheader("CPU")
        st.write(data['cpu'])
    with col2:
        st.subheader("Memory")
        st.write(data['memory'])
    with col3:
        st.subheader("GPU")
        st.write(data['gpu'])


def write_attr(root, name, data):
    data_len = len(data)
    if data_len > 0:
        col1, col2 = root.beta_columns([1, 9])
        with col1:
            st.subheader(str.upper(name))
        data_array = []
        for data_byte in data:
            data_str = data_byte.decode("utf-8")
            data_list = pd.Series(json.loads(data_str), dtype="float64")
            data_point = data_list.mean()
            data_array.append(data_point)

        data_len = len(data)

        chart_data = pd.DataFrame(data_array)
        chart_data['Time in minutes'] = chart_data.index
        if name in ["cpu", "gpu"]:
            chart_data[0] = chart_data[0] / 100
        alt_chart = alt.Chart(chart_data).mark_area().encode(
            x=alt.X('Time in minutes', type='quantitative'),
            y=alt.Y('0', scale=alt.Scale(domain=(0, 1)), axis=alt.Axis(
                format='%', title='Usage'), type='quantitative')
        )
        with col2:
            chart = st.altair_chart(alt_chart, use_container_width=True)
        return chart, chart_data, data_len
    else:
        raise Exception("No data yet...")


def write_resource_usage(root, name, data, dashboard_type):
    name_container = root.empty()
    node_ref = []

    if dashboard_type in [DashboardType.LOCAL]:
        col1, col2, col3 = root.beta_columns(3)
        with col1:
            st.subheader("CPU")
            node = st.empty()
            node_ref.append(node)
            node.write(data['cpu'])
        with col2:
            st.subheader("Memory")
            node = st.empty()
            node_ref.append(node)
            node.write(data['memory'])
        with col3:
            st.subheader("GPU")
            node = st.empty()
            node_ref.append(node)
            node.write(data['gpu'])
        name_container.header("Resource Remaining:")
    else:
        for attr_key in data:
            chart_ref, _, data_len = write_attr(root, attr_key, data[attr_key])
            node_ref.append(
                {'chart': chart_ref, 'len': data_len, 'chart_name': attr_key})
        name_container.header("Resource Usage:")
    return node_ref


def update_node(node_ref, new_data, dashboard_type, org_data_len):
    if dashboard_type in [DashboardType.LOCAL]:
        for attr_ref, att_name in zip(node_ref, ["cpu", "memory", "gpu"]):
            attr_ref.write(new_data[att_name])
    else:

        for attr_ref in node_ref:
            attr_data = new_data[attr_ref['chart_name']]
            data_len = len(attr_data)
            if data_len > 0:
                data_array = []
                for data_byte in attr_data:
                    data_str = data_byte.decode("utf-8")
                    data_list = pd.Series(json.loads(data_str), dtype="float64")
                    data_array.append(data_list.mean())
                chart_data = pd.DataFrame(data_array)
                chart_data['x'] = chart_data.index + org_data_len
                if attr_ref['chart_name'] in ["cpu", "gpu"]:
                    chart_data[0] = chart_data[0] / 100

                attr_ref['chart'].add_rows(chart_data)
                attr_ref['len'] += data_len


def write_job_list(root, list_name, list_data):
    root.subheader(f"{list_name}:")
    jqe = root.empty()
    update_job_list(jqe, list_name, list_data)
    return jqe


def update_job_list(root, list_name, list_data):
    if len(list_data) == 0:
        root.write(f"No job in queue \"{list_name}\" yet.")
    else:
        list_str = ["<ul>"]
        for data_byte in list_data:
            data_str = data_byte.decode("utf-8")
            list_str.append(f"<li>{data_str}</li>")
        list_str.append("</ul>")
        root.markdown("".join(list_str), True)


def write_job_queue(root, data):
    root.header("Job Queues:")
    pjqe = write_job_list(root, "Pending Jobs", data["pending_jobs"])
    kjqe = write_job_list(root, "Killed Jobs", data["killed_jobs"])
    return pjqe, kjqe


def update_job_queue(pjqe, kjqe, data):
    update_job_list(pjqe, "Pending Jobs", data["pending_jobs"])
    update_job_list(kjqe, "Killed Jobs", data["killed_jobs"])


def job_status_to_color(job_status):
    status_to_color = {
        JobStatus.RUNNING: "lightgreen",
        JobStatus.PENDING: "orange",
        JobStatus.KILLED: "gray",
        JobStatus.FINISH: "aqua",
        JobStatus.FAILED: "red",
    }
    return status_to_color[job_status]


def write_job_detail(data):

    component_html = """<table style="margin: 1rem; border_width: 0rem;">
<tr>
<th>Component</th> <th>Num</td> <th>Command</td>
</tr>"""

    for name in data['components']:
        component_html += f"""<tr>
<td>{name}</td> <td>{data['components'][name]['num']}</td> <td>{data['components'][name]['command']}</td>
</tr>"""

    component_html += """</table>"""

    color = job_status_to_color(data['status'])

    job_str = f"""<details>
<summary>
<span class="dot"
style="background-color: {color}; border-radius: 50%; height: 1rem; width: 1rem; display: inline-block;">
</span>
    {data['name']}    Mode: {data['mode']} </summary>
{component_html}
</details>"""
    return job_str
    # root.markdown(job_str, True)


def write_job_detail_container(data):

    component_html = """<table style="margin: 1rem; border_width: 0rem;">
<tr>
<th>Component</th>
<th>Num</td>
<th>Command</td>
<th>Image</th>
<th>CPU</td>
<th>Memory</td>
<th>GPU</th>
</tr>"""

    for name in data['components']:
        component_html += f"""<tr>
<td>{name}</td>
<td>{data['components'][name]['num']}</td>
<td>{data['components'][name]['command']}</td>
<td>{data['components'][name]['image']}</td>
<td>{data['components'][name]['resources']['cpu']}</td>
<td>{data['components'][name]['resources']['memory']}</td>
<td>{data['components'][name]['resources']['gpu']}</td>
</tr>"""

    component_html += """</table>"""

    color = job_status_to_color(data['status'])

    job_str = f"""<details>
<summary>
<span class="dot"
style="background-color: {color}; border-radius: 50%; height: 1rem; width: 1rem; display: inline-block;">
</span>
{data['name']}    Mode: {data['mode']}
</summary>
{component_html}
</details>"""
    return job_str
    # root.markdown(job_str, True)


def write_job_details(root, data, dashboard_type):
    root.header("Job Details:")
    jde = root.empty()
    update_job_details(jde, data, dashboard_type)
    return jde


def update_job_details(root, data, dashboard_type):
    if len(data) == 0:
        root.write("No job details yet.")
    else:
        list_str = ["<div>"]
        if dashboard_type is DashboardType.PROCESS:
            for job in data:
                list_str.append(write_job_detail(job))
        else:
            for job in data:
                list_str.append(write_job_detail_container(job))
        list_str.append("</div>")
        root.markdown("".join(list_str), True)


def load_executor(cluster_name):
    if cluster_name == "process":
        from maro.cli.process.executor import ProcessExecutor
        executor = ProcessExecutor()
        cluster_type = DashboardType.PROCESS
    else:
        from maro.cli.utils.details_reader import DetailsReader
        cluster_details = DetailsReader.load_cluster_details(
            cluster_name=cluster_name)
        if cluster_details["mode"] == "grass/azure":
            from maro.cli.grass.executors.grass_azure_executor import GrassAzureExecutor
            executor = GrassAzureExecutor(cluster_name=cluster_name)
            cluster_type = DashboardType.AZURE
        elif cluster_details["mode"] == "grass/on-premises":
            from maro.cli.grass.executors.grass_on_premises_executor import GrassOnPremisesExecutor
            executor = GrassOnPremisesExecutor(cluster_name=cluster_name)
            cluster_type = DashboardType.ONPREMISES
        elif cluster_details["mode"] == "grass/local":
            from maro.cli.grass.executors.grass_local_executor import GrassLocalExecutor
            executor = GrassLocalExecutor(cluster_name=cluster_name)
            cluster_type = DashboardType.LOCAL
    return executor, cluster_type


def load_clusters():
    clusters = []
    from maro.cli.utils.params import GlobalPaths
    for root, _, files in os.walk(GlobalPaths.ABS_MARO_CLUSTERS, topdown=False):
        for name in files:
            if os.path.basename(name) == "cluster_details.yml":
                clusters.append(os.path.basename(root))
    return clusters


def draw_dashboard_new(target):
    if not target:
        return

    try:
        static_node_load = False
        dynamic_node_load = False

        while True:

            if not (static_node_load or dynamic_node_load):
                local_executor, dashboard_type = load_executor(target)

                resource_data = local_executor.get_resource()

                # draw static content

                write_static_resurce(
                    st, resource_data, target, dashboard_type)

                st.markdown("---")

                static_node_load = True

                node_data_dict = local_executor.get_resource_usage(0)
                node_ref = write_resource_usage(
                    st, target, node_data_dict, dashboard_type)

                st.markdown("---")

                # jct = jce.beta_container()

                job_queue_data = local_executor.get_job_queue()
                pjqe, kjqe = write_job_queue(st, job_queue_data)

                st.markdown("---")

                job_detail_data = local_executor.get_job_details()
                jde = write_job_details(st, job_detail_data, dashboard_type)

                dynamic_node_load = True

            elif static_node_load and dynamic_node_load:
                data_len = node_ref[0]['len'] if dashboard_type != DashboardType.LOCAL else 0
                new_data_dict = local_executor.get_resource_usage(data_len)
                update_node(node_ref, new_data_dict, dashboard_type, data_len)

                job_queue_data = local_executor.get_job_queue()
                update_job_queue(pjqe, kjqe, job_queue_data)

                job_detail_data = local_executor.get_job_details()
                update_job_details(jde, job_detail_data, dashboard_type)

            time.sleep(5)
    except Exception as e:
        with st.spinner(f'No data yet...{traceback.format_exc()}'):
            time.sleep(5)


def main():
    target = None

    with st.spinner("Loading Clusters..."):
        clusters = load_clusters()
        for cluster_name in clusters:
            if st.sidebar.button(f'{cluster_name}'):
                target = cluster_name
    # with st.empty():
    draw_dashboard_new(target)


if __name__ == '__main__':
    main()
