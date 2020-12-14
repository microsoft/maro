
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import argparse
import crypt
import getpass
import os
import subprocess
import sys


# add user function
def create_user_with_promopt():

    user_name = input("Enter Username: ")
     # Asking for users password
    password = getpass.getpass()

    try:
        # executing useradd command using subprocess module
        subprocess.run(['useradd', '-p', password, user_name ])
    except:
        print(f"Failed to add user.")
        sys.exit(1)

def create_user(user_name: str, password: str):

    try:
        encPass = crypt.crypt(password, "22")
        os.system("useradd -p "+encPass+" johnsmith")
    except:
        print(f"Failed to add user.")
        sys.exit(1)


if __name__ == "__main__":
    # Load args
    parser = argparse.ArgumentParser()
    parser.add_argument('user_name')
    parser.add_argument('password', type=int)
    args = parser.parse_args()

    # create user
