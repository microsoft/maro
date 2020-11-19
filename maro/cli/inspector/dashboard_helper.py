import math

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from maro.cli.inspector.params import CIMItemOption, CITIBIKEOption, GlobalScenarios

# Pre-defined CSS style of inserted HTML elements.
title_html = """
<style>
    .title h1{
        font-size: 30px;
        color: black;
        background-size: 600vw 600vw;
        animation: slide 10s linear infinite forwards;
        margin-left:100px;
        text-align:left;
        margin-left:50px;
    }
    .title h3{
        font-size: 20px;
        color: grey;
        background-size: 600vw 600vw;
        animation: slide 10s linear infinite forwards;
        text-align:center
    }
    @keyframes slide {
        0%{
        background-position-x: 0%;
        }
        100%{
        background-position-x: 600vw;
        }
    }
</style>
"""


def render_h1_title(content: str):
    """Flexible display of content according to predefined styles.
    Args:
        content (str): Content to be showed on dashboard.
    """
    html_title = f"{title_html} <div class='title'><h1>{content}</h1></div>"
    st.markdown(html_title, unsafe_allow_html=True)


def render_h3_title(content: str):
    """Flexible display of content according to predefined styles.
    Args:
        content (str): Content to be showed on dashboard.
    """
    html_title = f"{title_html} <div class='title'><h3>{content}</h3></div>"
    st.markdown(html_title, unsafe_allow_html=True)


def get_holder_sample_ratio(snapshot_num: int) -> list:
    """Get sample data of holders.
    Condition: 1 must be included.
    Args:
        snapshot_num (int): Number of snapshots.

    Returns:
        list: sample data list
    """
    snapshot_sample_origin = round(1 / snapshot_num, 4)
    snapshot_sample_ratio = np.arange(snapshot_sample_origin, 1, snapshot_sample_origin).tolist()
    sample_ratio = [float("{:.4f}".format(i)) for i in snapshot_sample_ratio]
    if 1 not in sample_ratio:
        sample_ratio.append(1)

    return sample_ratio


def get_snapshot_sample_num(snapshot_num: int, snapshot_sample_num: float) -> list:
    """Get sample data of snapshot.
    Condition: 0 & 1 must be included.
    Args:
        snapshot_num (int): Number of snapshots.
        snapshot_sample_num (float): Expected number of sample data.

    Returns:
        list: snapshot sample data.
    """
    down_pooling = list(range(1, snapshot_num, math.floor(1 / snapshot_sample_num)))
    down_pooling.insert(0, 0)
    if snapshot_num - 1 not in down_pooling:
        down_pooling.append(snapshot_num - 1)
    return down_pooling


def get_filtered_formula_and_data(scenario: GlobalScenarios, data: pd.DataFrame, option_candidates: list) -> dict:
    """ Get calculated formula and whole data.

    Args:
        scenario (GlobalScenarios): Type of input scenario detail.
        data (dataframe): Original data.
        option_candidates (list): All options for users to choose.

    Returns:
        dict: calculated data, options.
    """
    data_generate = _formula_define(data)
    if data_generate is not None:
        data = data_generate["data"]
        option_candidates.append(data_generate["name"])

    item_option = st.multiselect(
        " ", option_candidates, option_candidates
    )
    item_option = _get_item_option(scenario, item_option, option_candidates)

    return {"data": data, "item_option": item_option}


@st.cache(allow_output_mutation=True)
def read_detail_csv(path: str) -> pd.DataFrame:
    """Read detail csv with cache.
    One thing to note: data is mutable.

    Args:
        path (str):  Path of file to be read.

    Returns:
        dataframe: Data in CSV file.

    """
    data = pd.read_csv(path)
    return data


def generate_by_snapshot_top_summary(
        attr_name: str, data: pd.DataFrame, top_number: int,
        attribute: str, snapshot_index: int = -1):
    """ Generate top-5 active holders and their summary data.

    Args:
        attr_name (str): Name of attributes needed to be summarized.
        data (dataframe): Data of selected column and snapshot_index.
        top_number (int): Number of top summaries.
        attribute (str): Attributes needed to be displayed on plot.
        snapshot_index (int): Index of snapshot.
    """
    if snapshot_index != -1:
        render_h3_title(f"SnapShot-{snapshot_index}:  Top {top_number} {attribute}")
    else:
        render_h3_title(f"Top {top_number} {attribute}")
    data = data[[attr_name, attribute]].sort_values(by=attribute, ascending=False).head(top_number)
    data["counter"] = range(len(data))
    data[attr_name] = list(map(lambda x, y: f"{x+1}-{y}", data["counter"], data[attr_name]))
    bars = alt.Chart(data).mark_bar().encode(
        x=attribute + ":Q",
        y=attr_name + ":O",
    ).properties(
        width=700,
        height=240
    )
    text = bars.mark_text(
        align="left",
        baseline="middle",
        dx=3
    ).encode(
        text=attribute + ":Q"
    )
    st.altair_chart(bars + text)


def _get_item_option(scenario: GlobalScenarios, item_option: list, option_candidates: list) -> list:
    """Convert selected CITI_BIKE option into column.

    Args:
        scenario (GlobalScenarios): Scenario name.
        item_option (list): User selected option list.
        option_candidates (list): Pre-defined option list.

    Returns:
        list: translated users" option.
    """
    item_option_res = []
    if scenario == GlobalScenarios.CITI_BIKE:
        for item in item_option:
            if item == "All":
                option_candidates.remove("All")
                option_candidates.remove("Requirement Info")
                option_candidates.remove("Station Info")
                item_option_res = option_candidates
                break
            elif item == "Requirement Info":
                item_option_res = item_option_res + CITIBIKEOption.requirement_info
            elif item == "Station Info":
                item_option_res = item_option_res + CITIBIKEOption.station_info
            else:
                item_option_res.append(item)
        return item_option_res

    if scenario == GlobalScenarios.CIM:
        for item in item_option:
            if item == "All":
                option_candidates.remove("All")
                option_candidates.remove("Booking Info")
                option_candidates.remove("Port Info")
                item_option_res = option_candidates
                break
            elif item == "Booking Info":
                item_option_res = item_option_res + CIMItemOption.booking_info
            elif item == "Port Info":
                item_option_res = item_option_res + CIMItemOption.port_info
            else:
                item_option_res.append(item)
        return item_option_res


def _formula_define(data_origin: pd.DataFrame) -> dict:
    """Define formula and get output
    Args:
        data_origin (dataframe): Data to be calculated.

    Returns:
        dict: formula name & formula output
    """
    st.sidebar.markdown("***")
    formula_select = st.sidebar.selectbox("formula:", ["a+b", "a-b", "a/b", "a*b", "sqrt(a)"])
    paras = st.sidebar.text_input("parameters separated by ;")
    res = paras.split(";")

    if formula_select == "a+b":
        if not _judge_data_length(res, 2):
            return
        else:
            data_right = _judge_append_data(data_origin.head(0), res)
            if data_right:
                data_origin[f"{res[0]}+{res[1]}"] = list(
                    map(lambda x, y: x + y,
                    data_origin[res[0]],
                    data_origin[res[1]]
                    )
                )
            else:
                return
        data = {"data": data_origin, "name": f"{res[0]}+{res[1]}"}
        return data

    if formula_select == "a-b":
        if not _judge_data_length(res, 2):
            return
        else:
            data_right = _judge_append_data(data_origin.head(0), res)
            if data_right:
                data_origin[f"{res[0]}-{res[1]}"] = list(
                    map(
                        lambda x, y: x - y,
                        data_origin[res[0]],
                        data_origin[res[1]]
                    )
                )
            else:
                return
        data = {"data": data_origin, "name": f"{res[0]}-{res[1]}"}
        return data

    if formula_select == "a*b":
        if not _judge_data_length(res, 2):
            return
        else:
            data_right = _judge_append_data(data_origin.head(0), res)
            if data_right:
                data_origin[f"{res[0]}*{res[1]}"] = list(
                    map(
                        lambda x, y: x * y,
                        data_origin[res[0]],
                        data_origin[res[1]]
                    )
                )
            else:
                return
        data = {"data": data_origin, "name": f"{res[0]}*{res[1]}"}
        return data

    if formula_select == "a/b":
        if not _judge_data_length(res, 2):
            return
        else:
            data_right = _judge_append_data(data_origin.head(0), res)
            if data_right:
                data_origin[f"{res[0]}/{res[1]}"] = list(
                    map(
                        lambda x, y: x + y,
                        data_origin[res[0]],
                        data_origin[res[1]]
                    )
                )
            else:
                return
        data = {"data": data_origin, "name": f"{res[0]}/{res[1]}"}
        return data

    if formula_select == "sqrt(a)":
        if not _judge_data_length(res, 1):
            return
        else:
            data_right = _judge_append_data(data_origin.head(0), res)
            if data_right:
                data_origin[f"sqrt({res[0]})"] = list(
                    map(lambda x: math.sqrt(x),
                        data_origin[res[0]]
                    )
                )
            else:
                return
        data = {"data": data_origin, "name": f"sqrt({res[0]})"}
        return data


def _judge_data_length(res: list, formula_length: int) -> bool:
    """ Judge whether the length of input data meet the requirements

    Args:
        res (list): Input data.
        formula_length (int): Supposed length of data list.

    Returns:
        bool: whether the length of data meet the requirements.
    """
    if len(res) == 0 or res[0] == "":
        return False
    elif len(res) != formula_length:
        st.warning("input parameter number wrong")
        return False
    return True


def _judge_append_data(data_head: list, res: list) -> bool:
    """Judge whether input is feasible to selected formula.

    Args:
        data_head (list): Column list of origin data.
        res (list): Column names texted by user.

    Returns:
        bool: Whether the column list texted by user is reasonable or not.

    """
    data_right = True
    for item in res:
        if item not in data_head:
            data_right = False
            st.warning(f"parameter name:{item} not exist")

    return data_right
