from .rl_component_bundle import rl_component_bundle as rcb
from maro.rl_v31.workflow.workflow import Workflow


def main() -> None:
    workflow = Workflow(
        rl_component_bundle=rcb,
        rollout_parallelism=1,  # TODO: config parallelism
        log_path="tests/rl_v31_gym/outputs/ppo",
    )
    workflow.train(
        num_iterations=1000,
        steps_per_iteration=4000,
        validation_interval=5,
        valid_episodes_per_iteration=10,
        checkpoint_interval=5,
        checkpoint_path="tests/rl_v31_gym/outputs/ppo/checkpoints",
    )


if __name__ == "__main__":
    main()
