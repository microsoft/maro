from typing import List

import numpy as np


def metrics_agg_func(metrics: List[dict]) -> dict:
    ret = {"n_interactions": sum(m["n_interactions"] for m in metrics)}
    metrics = [m for m in metrics if "n_steps" in m]
    if len(metrics) > 0:
        n_steps = sum(m["n_steps"] for m in metrics)
        n_segment = sum(m["n_segment"] for m in metrics)
        ret.update(
            {
                "n_steps": n_steps,
                "n_segment": n_segment,
                "avg_reward": np.sum([e["avg_reward"] * e["n_segment"] for e in metrics]) / n_segment,
                "avg_n_steps": np.sum([e["avg_n_steps"] * e["n_segment"] for e in metrics]) / n_segment,
                "max_n_steps": max(e["max_n_steps"] for e in metrics),
            },
        )

    return ret
