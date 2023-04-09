from .rl_component_bundle import rl_component_bundle as rcb
from maro.rl_v31.workflow.workflow import Workflow

def main() -> None:
    workflow = Workflow(
        rl_component_bundle=rcb,
        rollout_parallelism=1,  # TODO: config parallelism
        log_path="tests/rl_v31_gym/outputs/ppo",
    )
    workflow.train(
        num_iterations=30,
        steps_per_iteration=100,
        checkpoint_path="tests/rl_v31_gym/outputs/ppo/checkpoints",
        checkpoint_interval=3,
        validation_interval=3,
    )


if __name__ == "__main__":
    main()
