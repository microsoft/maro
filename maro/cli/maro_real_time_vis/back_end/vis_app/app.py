# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json

from data_process.request.request_attention import get_attention_data
from data_process.request.request_decision import get_acc_decision_data, get_decision_data
from data_process.request.request_exp_info import add_pending_experiment, get_experiment_info, get_experiments
from data_process.request.request_order import get_acc_order_data, get_order_data
from data_process.request.request_port import get_acc_port_data, get_new_port_number, get_port_data
from data_process.request.request_vessel import get_acc_vessel_data, get_vessel_data
from flask import Flask, jsonify, request
from flask_cors import CORS

app_backend = Flask(__name__)
CORS(app_backend, supports_credentials=True)


@app_backend.route('/get_latest_epoch', methods=['GET', 'POST'])
def get_new_epoch() -> str:
    """Get the latest epoch number.

    Returns:
        str: Epoch number.

    """
    data = request.get_json(silent=True)
    cur_experiment_name = data["experiment_name"]
    return get_new_port_number(cur_experiment_name)


@app_backend.route('/experiments', methods=['GET', 'POST'])
def get_existing_experiments() -> str:
    """Get a list of existing experiments.

    Returns:
        str: Experiments information.

    """
    return get_experiments()


@app_backend.route('/pending_experiment', methods=['GET', 'POST'])
def add_experiment() -> str:
    """Add a pending experiment.

    Returns:
        str: Experiment information.

    """
    data = request.get_json(silent=True)
    cur_experiment_name = data["experiment_name"]
    return add_pending_experiment(cur_experiment_name)


@app_backend.route('/experiment_info', methods=['GET', 'POST'])
def get_basic_experiment_info() -> str:
    """Get basic experiment information.

    Returns:
        str: Experiment information.

    """
    return get_experiment_info()


@app_backend.route('/get_snapshot_vessel', methods=['GET', 'POST'])
def get_basic_vessel_info() -> str:
    """Get vessel information within one tick.

    Returns:
        str: Vessel information.

    """
    data = request.get_json(silent=True)
    cur_experiment_name = data["experiment_name"]
    cur_snapshot_number = data["snapshot_number"]
    cur_epoch_number = data["epoch_number"]
    vessel_data = get_vessel_data(
        cur_experiment_name, cur_epoch_number, cur_snapshot_number
    )
    return vessel_data


@app_backend.route('/get_acc_snapshot_vessel', methods=['GET', 'POST'])
def get_acc_vessel_info() -> str:
    """Get vessel information within a range.

    Returns:
        str: Vessel information.

    """
    data = request.get_json(silent=True)
    cur_experiment_name = data["experiment_name"]
    snapshot_start_number = data["snapshot_start_number"]
    snapshot_end_number = data["snapshot_end_number"]
    cur_epoch_number = data["epoch_number"]
    vessel_data = get_acc_vessel_data(
        cur_experiment_name, cur_epoch_number, snapshot_start_number, snapshot_end_number
    )
    return vessel_data


@app_backend.route('/get_snapshot_port', methods=['GET', 'POST'])
def get_basic_port_info() -> str:
    """Get port information within one tick.

    Returns:
        str: Port information.

    """
    data = request.get_json(silent=True)
    cur_experiment_name = data["experiment_name"]
    cur_snapshot_number = data["snapshot_number"]
    cur_epoch_number = data["epoch_number"]
    port_data = get_port_data(cur_experiment_name, cur_epoch_number, cur_snapshot_number)
    return port_data


@app_backend.route('/get_acc_snapshot_port', methods=['GET', 'POST'])
def get_acc_port_info() -> str:
    """Get vessel information within a range.

    Returns:
        str: Port information.

    """
    data = request.get_json(silent=True)
    cur_experiment_name = data["experiment_name"]
    snapshot_start_number = data["snapshot_start_number"]
    snapshot_end_number = data["snapshot_end_number"]
    cur_epoch_number = data["epoch_number"]
    port_data = get_acc_port_data(
        cur_experiment_name, cur_epoch_number, snapshot_start_number, snapshot_end_number
    )
    return port_data


@app_backend.route('/get_snapshot_order', methods=['GET', 'POST'])
def get_basic_order_info() -> str:
    """Get order information within one tick.

    Returns:
        str: Order information.

    """
    data = request.get_json(silent=True)
    cur_experiment_name = data["experiment_name"]
    cur_snapshot_number = data["snapshot_number"]
    cur_epoch_number = data["epoch_number"]
    order_data = get_order_data(cur_experiment_name, cur_epoch_number, cur_snapshot_number)
    return order_data


@app_backend.route('/get_acc_snapshot_order', methods=['GET', 'POST'])
def get_acc_order_info() -> str:
    """Get order information within a range.

    Returns:
        str: Order information.

    """
    data = request.get_json(silent=True)
    cur_experiment_name = data["experiment_name"]
    snapshot_start_number = data["snapshot_start_number"]
    snapshot_end_number = data["snapshot_end_number"]
    cur_epoch_number = data["epoch_number"]
    order_data = get_acc_order_data(cur_experiment_name, cur_epoch_number, snapshot_start_number, snapshot_end_number)
    return jsonify(order_data)


@app_backend.route('/get_snapshot_attention', methods=['GET', 'POST'])
def get_basic_attention_info() -> str:
    """Get attention information within one tick.

    Returns:
        str: Attention information.

    """
    data = request.get_json(silent=True)
    cur_experiment_name = data["experiment_name"]
    cur_snapshot_number = data["snapshot_number"]
    cur_epoch_number = data["epoch_number"]
    attention_data = get_attention_data(cur_experiment_name, cur_epoch_number, cur_snapshot_number)
    return attention_data


@app_backend.route('/get_snapshot_decision', methods=['GET', 'POST'])
def get_basic_decision_info() -> str:
    """Get decision information within one tick.

    Returns:
        str: Decision information.

    """
    data = request.get_json(silent=True)
    cur_experiment_name = data["experiment_name"]
    cur_snapshot_number = data["snapshot_number"]
    cur_epoch_number = data["epoch_number"]
    decision_data = get_decision_data(cur_experiment_name, cur_epoch_number, cur_snapshot_number)
    return decision_data


@app_backend.route('/get_acc_snapshot_decision', methods=['GET', 'POST'])
def get_acc_decision_info() -> str:
    """Get vessel information within a range.

    Returns:
        str: Decision information.

    """
    data = request.get_json(silent=True)
    cur_experiment_name = data["experiment_name"]
    snapshot_start_number = data["snapshot_start_number"]
    snapshot_end_number = data["snapshot_end_number"]
    cur_epoch_number = data["epoch_number"]
    decision_data = get_acc_decision_data(
        cur_experiment_name, cur_epoch_number, snapshot_start_number, snapshot_end_number
    )
    return jsonify(decision_data)


@app_backend.route('/get_acc_attrs', methods=['GET', 'POST'])
def get_acc_attrs():
    """Get decision and order information within a range.

    Returns:
        str: Decision and order information.

    """
    data = request.get_json(silent=True)
    cur_experiment_name = data["experiment_name"]
    snapshot_start_number = data["snapshot_start_number"]
    snapshot_end_number = data["snapshot_end_number"]
    cur_epoch_number = data["epoch_number"]
    decision_data = get_acc_decision_data(
        cur_experiment_name, cur_epoch_number, snapshot_start_number, snapshot_end_number
    )
    order_data = get_acc_order_data(cur_experiment_name, cur_epoch_number, snapshot_start_number, snapshot_end_number)
    output = {"decision": decision_data, "order": order_data}
    return json.dumps(output)


# Use Only For Local Debug
# ************************************
if __name__ == '__main__':
    app_backend.run(debug=True, port=5000)
# ************************************
