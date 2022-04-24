import os


def extract_trainer_name(policy_name: str) -> str:
    """Extract the trainer name from the policy name.

    Args:
        policy_name (str): Policy name.

    Returns:
        trainer_name (str)
    """
    return policy_name.split(".")[0]


def get_latest_ep(path: str) -> int:
    ep_list = [int(ep) for ep in os.listdir(path)]
    return max(ep_list)
