# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import argparse
import crypt
import getpass
import os
import subprocess
import sys

"""
This file is used for deleting a specified account and related files on node.
Example:
sudo python3 delete_user.py {account name}
"""

def delete_user(user_name: str):

    try:
        user_path = "/home/" + user_name
        os.system("sudo userdel " + user_name)
        if os.path.exists(user_path):
            os.system("sudo rm -rf " + user_path)
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
