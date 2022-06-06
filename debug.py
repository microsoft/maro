import sys
sys.path.append(".")

from maro.cli.local.commands import run

if __name__ == "__main__":
     run(conf_path=r"examples\\rl\\supply_chain.yml", containerize=False, evaluate_only=True)
