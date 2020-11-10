import math
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# Pre-defined CSS style of inserted HTML elements.
Title_html = """
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


def get_epoch_num(origin_len, ROOT_PATH):
    """Get epoch num by counting folders start with "snapshot_"

    Args:
        origin_len(int): Length of all file&folder under root path.
        ROOT_PATH(str):  Data folder path.

    Returns:
        int: Number of epoches.
    """
    epoch_num = 0
    for index in range(0, origin_len):
        if os.path.exists(os.path.join(ROOT_PATH, r"snapshot_" + str(index))):
            epoch_num = epoch_num + 1
    return epoch_num


def render_H1_title(content):
    """Flexible display of content according to predefined styles.
    Args:
        content(str): Content to be showed on dashboard.
    """
    html_title = Title_html + "<div class='title'><h1>" + content + "</h1></div>"
    st.markdown(html_title, unsafe_allow_html=True)


def render_H3_title(content):
    """Flexible display of content according to predefined styles.
    Args:
        content(str): Content to be showed on dashboard.
    """
    html_title = Title_html + "<div class='title'><h3>" + content + "</h3></div>"
    st.markdown(html_title, unsafe_allow_html=True)


def holder_sample_ratio(snapshot_num):
    """Get sample data of holders.
    Condition: 1 must be included.
    Args:
        snapshot_num(int): Number of snapshots.

    Returns:
        list: sample data list

    """
    snapshot_sample_origin = round(1 / snapshot_num, 4)
    snapshot_sample_ratio = np.arange(snapshot_sample_origin, 1, snapshot_sample_origin).tolist()
    sample_ratio = [float("{:.4f}".format(i)) for i in snapshot_sample_ratio]
    if 1 not in sample_ratio:
        sample_ratio.append(1)
    return sample_ratio


def get_snapshot_sample(snapshot_num, snapshot_sample_num):
    """Get sample data of snapshot.
    Condition: 0 & 1 must be included.
    Args:
        snapshot_num(int): Number of snapshots.
        snapshot_sample_num(int): Expected number of sample data.

    Returns:
        list: snapshot sample data.

    """
    down_pooling = list(range(1, snapshot_num, math.floor(1 / snapshot_sample_num)))
    down_pooling.insert(0, 0)
    if snapshot_num - 1 not in down_pooling:
        down_pooling.append(snapshot_num - 1)
    return down_pooling


def formula_define(data_origin):
    """Define formula and get output
    Args:
        data_origin(list): Data to be calculated.

    Returns:
        dict: formula name & formula output
    """
    st.sidebar.markdown("***")
    formula_select = st.sidebar.selectbox("formula:", ["a+b", "a*b+sqrt(c*d)"])
    paras = st.sidebar.text_input("parameters separated by ;")
    res = paras.split(";")
    if formula_select == "a+b":
        if len(res) == 0 or res[0] == "":
            return
        elif len(res) != 2:
            st.warning("input parameter number wrong")
            return
        else:
            data_right = judge_append_data(data_origin.head(0), res)
            if data_right:
                data_origin[res[0] + "+" + res[1]] = list(
                    map(lambda x, y: x + y, data_origin[res[0]], data_origin[res[1]]))
            else:
                return
        data = {"data_after": data_origin, "name": res[0] + "+" + res[1]}
        return data
    if formula_select == "a*b+sqrt(c*d)":
        if len(res) == 0 or res[0] == "":
            return
        elif len(res) != 4:
            st.warning("input parameter number wrong")
            return
        else:
            data_right = judge_append_data(data_origin.head(0), res)
            if data_right:
                data_origin[res[0] + "*" + res[1] + "+sqrt(" + res[2] + "*+" + res[3] + ")"] = list(
                    map(lambda x, y, z, w: int(x) * int(y) + math.sqrt(z * int(w)),
                        data_origin[res[0]], data_origin[res[1]], data_origin[res[2]], data_origin[res[3]]))
            else:
                return
        data = {"data_after": data_origin, "name": res[0] + "*" + res[1] + "+sqrt(" + res[2] + "*+" + res[3] + ")"}
        return data


def judge_append_data(data_head, res):
    """Judge whether input is feasible to selected formula.

    Args:
        data_head(list): Column list of origin data.
        res: Column names texted by user.

    Returns:
        bool: Whether the column list texted by user is reasonable or not.

    """
    data_right = True
    for item in res:
        if item not in data_head:
            data_right = False
            st.warning("parameter name:" + item + " not exist")
    return data_right


@st.cache(allow_output_mutation=True)
def read_detail_csv(path):
    data = pd.read_csv(path)
    return data


def generate_by_snapshot_top_summary(attr_name, data, attribute, Need_SnapShot, snapshot_index=-1):
    """ Generate top-5 active holders and their summary data.

    Args:
        attr_name(str): Name of attributes needed to be summarized.
        data(list): Data of selected column and snapshot_index.
        attribute(str): Attributes needed to be displayed on plot.
        Need_SnapShot(bool): Have a snapshot index or not.
        snapshot_index(int): Index of snapshot.
    """
    if Need_SnapShot:
        render_H3_title("SnapShot-" + str(snapshot_index) + ": " + "     Top 5 " + attribute)
    else:
        render_H3_title("Top 5 " + attribute)
    data["counter"] = range(len(data))
    data[attr_name] = list(map(lambda x, y: str(x + 1) + "-" + y, data["counter"], data[attr_name]))
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
