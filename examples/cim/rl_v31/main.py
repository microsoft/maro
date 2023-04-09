from examples.cim.rl_v31.rl_component_bundle import rcb
from maro.rl_v31.workflow.workflow import Workflow

PARALLELISM = 1


def main() -> None:
    workflow = Workflow(
        rl_component_bundle=rcb,
        rollout_parallelism=PARALLELISM,
        log_path="examples/cim/rl_v31/log/",
    )
    workflow.train(
        num_iterations=30,
        episodes_per_iteration=1,
        checkpoint_path="examples/cim/rl_v31/log/checkpoints/",
        checkpoint_interval=3,
        validation_interval=3,
        # early_stop_config=("container_shortage", 2, False),
    )


if __name__ == "__main__":
    main()
