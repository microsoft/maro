# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import argparse
import crypt
import getpass
import os
import subprocess
import sys

"""
This file is used for creating a user account with SSH public key settings on node.
Example:
sudo python3 create_user.py {account name} "{RSA public key}"
"""

def create_user(user_name: str) -> None:
    try:
        os.system("sudo useradd -m " + user_name)
        os.system("sudo usermod -G root " + user_name)
        ssh_path = f"/home/{user_name}/.ssh/"
        if not os.path.exists(ssh_path):
            os.mkdir(ssh_path)
    except:
        print("Failed to add user.")
        sys.exit(1)

def add_pub_key(user_name: str, pub_key: str) -> None:
    ssh_path = f"/home/{user_name}/.ssh/"
    authorized_keys_path = os.path.join(ssh_path, "authorized_keys")
    with open(authorized_keys_path, "w+") as pub_key_file:
        lines = ["\r\n", pub_key, "\r\n"]
        pub_key_file.writelines(lines)
        pub_key_file.close()

# Please don't test on your own macOS or Linux.
# Sudoers file doesn't accept "\r\n", but only "\r" seems OK.
def add_sudoers(user_name: str) -> None:
    account_line = f"{user_name}    ALL=(ALL:ALL) NOPASSWD:ALL"
    with open("/etc/sudoers", "a+") as sudoers_file:
        lines = [account_line]
        sudoers_file.writelines(lines)
        sudoers_file.close()

def check_sudoers(user_name: str) -> bool:
    account_line = f"{user_name}    ALL=(ALL:ALL) NOPASSWD:ALL"
    with open("/etc/sudoers", "r") as sudoers_file:
        lines = sudoers_file.readlines()
        sudoers_file.close()

    for line in lines:
        if account_line in line:
            return True

    return False

def user_already_exists(user_name: str) -> bool:
    user_path = "/home/" + user_name
    if os.path.exists(user_path):
        return True

    return False

if __name__ == "__main__":
    # Load args
    parser = argparse.ArgumentParser()
    parser.add_argument("user_name")
    parser.add_argument("pub_key")
    args = parser.parse_args()

    if not user_already_exists(args.user_name):
        # create user
        create_user(args.user_name)
        user_path = "/home/" + args.user_name
        os.system(f"sudo ssh-keygen -t rsa -N '' -f {user_path}/.ssh/id_rsa")
        if not check_sudoers(args.user_name):
            add_sudoers(args.user_name)
    # set pub key
    add_pub_key(args.user_name, args.pub_key)
