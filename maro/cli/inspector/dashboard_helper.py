# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
from typing import List

import altair as alt
import pandas as pd
import streamlit as st

from maro.cli.inspector.params import CIMItemOption, CITIBIKEItemOption, GlobalScenarios

# Pre-defined CSS style of inserted HTML elements.
title_html = """
<style>
    .title h1{
        font-size: 30px;
        color: black;
        background-size: 600vw 600vw;
        animation: slide 10s linear infinite forwards;
        margin-left: 100px;
        text-align: left;
        margin-left: 50px;
    }
    .title h3{
        font-size: 20px;
        color: grey;
        background-size: 600vw 600vw;
        animation: slide 10s linear infinite forwards;
        text-align: center
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
        Content (str): Content to be showed on dashboard.
    """
    html_title = f"{title_html} <div class='title'><h1>{content}</h1></div>"
    st.markdown(html_title, unsafe_allow_html=True)


def render_h3_title(content: str):
    """Flexible display of content according to predefined styles.

    Args:
        Content (str): Content to be showed on dashboard.
    """
    html_title = f"{title_html} <div class='title'><h3>{content}</h3></div>"
    st.markdown(html_title, unsafe_allow_html=True)


def get_sample_ratio_selection_list(data_num: int) -> list:
    """Get sample ratio list with current data_num.

    Note: Number 0,1 must be included in the return value.
    Because it is essential for user to select sample ratio at 0/1.

    Args:
        data_num (int): Number of data to be sampled.

    Returns:
        List(float): The sample ratio list for user to select, which is a list range from 0 to 1.

    """
    return [round(i / data_num, 2) for i in range(0, data_num + 1)]


def get_sample_index_list(data_num: int, sample_ratio: float) -> List[float]:
    """Get the index list of sampled data.

    Args:
        data_num (int): Number of data to be sampled.
        sample_ratio (float): The sample ratio selected by user.

    Returns:
        List[float]: The list of sampled data index list, which is a list range from 0 to data_num.

    """
    if sample_ratio == 0:
        return []
    else:
        return list(range(0, data_num, math.floor(1 / sample_ratio)))


def get_filtered_formula_and_data(
    scenario: GlobalScenarios, data: pd.DataFrame, attribute_option_candidates: List[str]
) -> dict:
    """ Get calculated formula and whole data.

    Args:
        scenario (GlobalScenarios): Type of input scenario detail.
        data (pd.Dataframe): Original data.
        attribute_option_candidates (List[str]): All options for users to choose.

    Returns:
        dict: The data and attributes after formula calculation.

    """
    data_generate = _formula_define(data)
    if data_generate is not None:
        data = data_generate["data"]
        attribute_option_candidates.append(data_generate["name"])

    attribute_option = st.multiselect(
        label=" ",
        options=attribute_option_candidates,
        default=attribute_option_candidates
    )
    attribute_option = _get_attribute_option(scenario, attribute_option, attribute_option_candidates)
    return {"data": data, "attribute_option": attribute_option}


@st.cache(allow_output_mutation=True)
def read_detail_csv(path: str) -> pd.DataFrame:
    """Read detail csv with cache.

    Args:
        path (str): Path of file to be read.

    Returns:
        pd.Dataframe: Data in CSV file. Cached in local memory.

    """
    return pd.read_csv(path)


def generate_by_snapshot_top_summary(
    attr_name: str, data: pd.DataFrame, top_number: int,
    attribute: str, snapshot_index: int = -1
):
    """ Generate top-k resource holders and their according summary data.

    Args:
        attr_name (str): Name of attributes needed to be summarized.
        data (pd.Dataframe): Data of selected column and snapshot_index.
        top_number (int): Number of top summaries.
        attribute (str): Attributes needed to be displayed on plot.
        snapshot_index (int): Index of snapshot.
    """
    if snapshot_index != -1:
        render_h3_title(f"Snapshot-{snapshot_index}: Top {top_number} {attribute}")
    else:
        render_h3_title(f"Top {top_number} {attribute}")
    data = data[[attr_name, attribute]].sort_values(by=attribute, ascending=False).head(top_number)
    data["counter"] = range(len(data))
    data[attr_name] = list(
        map(
            lambda x, y: f"{x+1}-{y}",
            data["counter"],
            data[attr_name]
        )
    )
    bars = alt.Chart(data).mark_bar().encode(
        x=f"{attribute}:Q",
        y=f"{attr_name}:O",
    ).properties(
        width=700,
        height=240
    )
    text = bars.mark_text(
        align="left",
        baseline="middle",
        dx=3
    ).encode(
        text=f"{attribute}:Q"
    )
    st.altair_chart(bars + text)


def _get_attribute_option(
    scenario: GlobalScenarios, attribute_option: List[str], attribute_option_candidates: List[str]
) -> List[str]:
    """Convert selected attribute options into column.

    Args:
        scenario (GlobalScenarios): Name of scenario
        attribute_option (List[str]): User-selected attributes list.
        attribute_option_candidates (List[str]): Pre-defined attributes list.

    Returns:
        List[str]: Translated attribute list,
            which contains no user-selected attributes, but only pre-defined attributes.

    """
    attribute_option_res = []
    if scenario == GlobalScenarios.CITI_BIKE:
        for item in attribute_option:
            if item == "All":
                attribute_option_candidates.remove("All")
                attribute_option_candidates.remove("Requirement Info")
                attribute_option_candidates.remove("Station Info")
                attribute_option_res = attribute_option_candidates
                break
            elif item == "Requirement Info":
                attribute_option_res = attribute_option_res + CITIBIKEItemOption.requirement_info
            elif item == "Station Info":
                attribute_option_res = attribute_option_res + CITIBIKEItemOption.station_info
            else:
                attribute_option_res.append(item)
        return attribute_option_res

    if scenario == GlobalScenarios.CIM:
        for item in attribute_option:
            if item == "All":
                attribute_option_candidates.remove("All")
                attribute_option_candidates.remove("Booking Info")
                attribute_option_candidates.remove("Port Info")
                attribute_option_res = attribute_option_candidates
                break
            elif item == "Booking Info":
                attribute_option_res = attribute_option_res + CIMItemOption.booking_info
            elif item == "Port Info":
                attribute_option_res = attribute_option_res + CIMItemOption.port_info
            else:
                attribute_option_res.append(item)
        return attribute_option_res


def _formula_define(data_original: pd.DataFrame) -> dict:
    """Define formula and get output.

    Args:
        data_original (pd.Dataframe): Original data to be calculated with selected formula.

    Returns:
        dict: Name and output of selected formula and original data.

    """
    st.sidebar.markdown("***")
    formula_select = st.sidebar.selectbox(
        label="formula:",
        options=["a+b", "a-b", "a/b", "a*b", "sqrt(a)"]
    )
    paras = st.sidebar.text_input("parameters separated by ;")
    res = paras.split(";")

    if formula_select == "a+b":
        if not _judge_data_length(res, 2):
            return
        else:
            data_right = _judge_append_data(data_original.head(0), res)
            if data_right:
                data_original[f"{res[0]}+{res[1]}"] = list(
                    map(
                        lambda x, y: x + y,
                        data_original[res[0]],
                        data_original[res[1]]
                    )
                )
            else:
                return
        data = {"data": data_original, "name": f"{res[0]}+{res[1]}"}
        return data

    if formula_select == "a-b":
        if not _judge_data_length(res, 2):
            return
        else:
            data_right = _judge_append_data(data_original.head(0), res)
            if data_right:
                data_original[f"{res[0]}-{res[1]}"] = list(
                    map(
                        lambda x, y: x - y,
                        data_original[res[0]],
                        data_original[res[1]]
                    )
                )
            else:
                return
        data = {"data": data_original, "name": f"{res[0]}-{res[1]}"}
        return data

    if formula_select == "a*b":
        if not _judge_data_length(res, 2):
            return
        else:
            data_right = _judge_append_data(data_original.head(0), res)
            if data_right:
                data_original[f"{res[0]}*{res[1]}"] = list(
                    map(
                        lambda x, y: x * y,
                        data_original[res[0]],
                        data_original[res[1]]
                    )
                )
            else:
                return
        data = {"data": data_original, "name": f"{res[0]}*{res[1]}"}
        return data

    if formula_select == "a/b":
        if not _judge_data_length(res, 2):
            return
        else:
            data_right = _judge_append_data(data_original.head(0), res)
            if data_right:
                data_original[f"{res[0]}/{res[1]}"] = list(
                    map(
                        lambda x, y: x + y,
                        data_original[res[0]],
                        data_original[res[1]]
                    )
                )
            else:
                return
        data = {"data": data_original, "name": f"{res[0]}/{res[1]}"}
        return data

    if formula_select == "sqrt(a)":
        if not _judge_data_length(res, 1):
            return
        else:
            data_right = _judge_append_data(data_original.head(0), res)
            if data_right:
                data_original[f"sqrt({res[0]})"] = list(
                    map(
                        lambda x: math.sqrt(x),
                        data_original[res[0]]
                    )
                )
            else:
                return
        data = {"data": data_original, "name": f"sqrt({res[0]})"}
        return data


def _judge_data_length(res: list, formula_length: int) -> bool:
    """Judge whether the length of input data meet the requirements.

    Args:
        res (list): User-input data.
        formula_length (int): Supposed length of input data with selected formula.

    Returns:
        bool: Whether the length of data meet the requirements of formula.

    """
    if len(res) == 0 or res[0] == "":
        return False
    elif len(res) != formula_length:
        st.warning("input parameter number wrong")
        return False
    return True


def _judge_append_data(data_head: List[str], res: List[str]) -> bool:
    """Judge whether input is feasible to selected formula.

    Args:
        data_head (List[str]): Column list of origin data.
        res (List[str]): Column names texted by user.

    Returns:
        bool: Whether the column list texted by user is reasonable or not.

    """
    data_right = True
    for item in res:
        if item not in data_head:
            data_right = False
            st.warning(f"parameter name:{item} not exist")

    return data_right


def _get_sampled_epoch_range(epoch_num: int, sample_ratio: float) -> List[float]:
    """For inter plot, generate sampled data list based on range & sample ratio.

    Args:
        epoch_num (int): Total number of epoches,
            i.e. the total number of data folders since there is a folder per epoch.
        sample_ratio (float): Sampling ratio.
            e.g. If sample_ratio = 0.3, and sample data range = [0, 10],
            down_pooling_list = [0, 0.3, 0.6, 0.9]
            down_pooling_range = [0, 3, 6, 9]

    Returns:
        List[float]: List of sampled epoch index.

    """
    start_epoch = st.sidebar.number_input(
        label="Start Epoch",
        min_value=0,
        max_value=epoch_num - 1,
        value=0
    )
    end_epoch = st.sidebar.number_input(
        label="End Epoch",
        min_value=0,
        max_value=epoch_num - 1,
        value=epoch_num - 1
    )
    down_pooling_num = st.sidebar.select_slider(
        label="Epoch Sampling Ratio",
        options=sample_ratio,
        value=1
    )
    if down_pooling_num == 0:
        return []
    else:
        return list(range(start_epoch, end_epoch + 1, math.floor(1 / down_pooling_num)))
