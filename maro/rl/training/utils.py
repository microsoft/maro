# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

def extract_trainer_name(policy_name: str) -> str:
    return policy_name.split(".")[0]
