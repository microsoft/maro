# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import argparse
import crypt
import getpass
import os
import platform
import subprocess
import sys

"""
This file is used for deleting a specified account and related files on node.
Example:
sudo python3 delete_user.py {account name}
"""


def run_command(command: str) -> str:
    if platform.system() == "Windows":
        command = f"powershell.exe -Command \"{command}\""

    completed_process = subprocess.run(
        command,
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )
    if completed_process.returncode != 0:
        return completed_process.stderr
    return completed_process.stdout

def delete_user(user_name: str):

    try:
        user_path = "/home/" + user_name
        run_command("sudo userdel -f " + user_name)
        if os.path.exists(user_path):
            run_command("sudo rm -rf " + user_path)
    except:
        print("Failed to delete user.")
        sys.exit(1)

def user_already_exists(user_name: str) -> bool:
    user_path = "/home/" + user_name
    if os.path.exists(user_path):
        return True

    return False

if __name__ == "__main__":
    # Load args
    parser = argparse.ArgumentParser()
    parser.add_argument("user_name")
    args = parser.parse_args()

    if user_already_exists(args.user_name):
        # delete user
        delete_user(args.user_name)
        print(f"The account {args.user_name} has been deleted.")
    else:
        print(f"The account {args.user_name} does not exists.")
