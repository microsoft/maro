import copy
import random
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt


class Trainer(object):
    """Runs games for given agent. Optionally will visualise and save the results"""
    def __init__(self, config, agent_class, env):
        self.config = config
        self.env = env
        self.agent_class = agent_class
        self.results = None
        self.colour_ix = 0

    def load_model_for_agent(self):
        agent_config = copy.deepcopy(self.config)
        agent = self.agent_class(agent_config, self.env)
        agent.load_local_critic()
        return agent

    def _create_object_to_store_results(self):
        """Creates a dictionary that we will store the results in if it doesn't exist, otherwise it loads it up"""
        if self.config.overwrite_existing_results_file \
            or not self.config.file_to_save_data_results \
            or not os.path.isfile(self.config.file_to_save_data_results):
            results = []
        else:
            with open(self.config.file_to_save_data_results, 'rb') as f:
                results = pickle.load(f)
        return results

    def run_games_for_agent(self):
        print("\033[1m" + "AGENT NAME: {}".format(self.agent_class.agent_name) + "\033[0m", flush=True)

        self.results = self._create_object_to_store_results()

        agent_config = copy.deepcopy(self.config)
        if self.config.randomise_random_seed:
            agent_config.seed = random.randint(0, 2**32 - 2)
        self.env.reset()
        agent = self.agent_class(agent_config, self.env)

        print(agent.hyperparameters)
        print("RANDOM SEED " , agent_config.seed)

        game_scores, rolling_scores, time_taken = agent.run_n_episodes()
        print("Time taken: {}".format(time_taken), flush=True)

        self.results.append([game_scores, rolling_scores, len(rolling_scores), -1 * max(rolling_scores), time_taken])

        self.visualise_overall_agent_results(
            agent.environment_title, [rolling_scores], self.agent_class.agent_name, show_mean_and_std_range=False
        )
        self._save_obj(self.results, self.config.file_to_save_data_results)
        plt.savefig(self.config.file_to_save_results_graph, bbox_inches="tight")
        return self.results

    def visualise_overall_agent_results(
        self, environment_name, agent_results, agent_name, show_mean_and_std_range=False
    ):
        """Visualises the results for one agent"""
        assert isinstance(agent_results, list), "agent_results must be a list of lists, 1 set of results per list"
        assert isinstance(agent_results[0], list), "agent_results must be a list of lists, 1 set of results per list"
        ax = plt.gca()
        color = "#800000"
        if show_mean_and_std_range:
            mean_minus_x_std, mean_results, mean_plus_x_std = self._get_mean_and_standard_deviation_difference_results(agent_results)
            x_vals = list(range(len(mean_results)))
            ax.plot(x_vals, mean_results, label=agent_name, color=color)
            ax.plot(x_vals, mean_plus_x_std, color=color, alpha=0.1)
            ax.plot(x_vals, mean_minus_x_std, color=color, alpha=0.1)
            ax.fill_between(x_vals, y1=mean_minus_x_std, y2=mean_plus_x_std, alpha=0.1, color=color)
        else:
            for ix, result in enumerate(agent_results):
                x_vals = list(range(len(agent_results[0])))
                plt.plot(x_vals, result, label=agent_name + "_{}".format(ix+1), color=color)
                color = self._get_next_color()

        ax.set_facecolor('xkcd:white')

        # Shrink current axis's height by 10% on the bottom
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.05, box.width, box.height * 0.95])

        # Put a legend below current axis
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=3)

        ax.set_title(environment_name, fontsize=15, fontweight='bold')
        ax.set_ylabel('Rolling Episode Scores')
        ax.set_xlabel('Episode Number')
        self._hide_spines(ax, ['right', 'top'])
        ax.set_xlim([0, x_vals[-1]])

        y_min, y_max = self._get_y_limits(agent_results)

        ax.set_ylim([y_min, y_max])

    def _get_y_limits(self, results):
        """Extracts the minimum and maximum seen y_values from a set of results"""
        min_result = float("inf")
        max_result = float("-inf")
        for result in results:
            temp_max = np.max(result)
            temp_min = np.min(result)
            if temp_max > max_result:
                max_result = temp_max
            if temp_min < min_result:
                min_result = temp_min
        return min_result, max_result

    def _get_next_color(self):
        """Gets the next color in list self.colors. If it gets to the end then it starts from beginning"""
        colors = ["red", "blue", "green", "orange", "yellow", "purple"]
        self.colour_ix += 1
        if self.colour_ix >= len(colors): self.colour_ix = 0
        color = colors[self.colour_ix]
        return color

    def _get_mean_and_standard_deviation_difference_results(self, results):
        """From a list of lists of agent results it extracts the mean results and the mean results plus or minus
         some multiple of the standard deviation"""
        def get_results_at_a_time_step(results, timestep):
            results_at_a_time_step = [result[timestep] for result in results]
            return results_at_a_time_step
        def get_standard_deviation_at_time_step(results, timestep):
            results_at_a_time_step = [result[timestep] for result in results]
            return np.std(results_at_a_time_step)
        mean_results = [np.mean(get_results_at_a_time_step(results, timestep)) for timestep in range(len(results[0]))]
        mean_minus_x_std = [mean_val - self.config.standard_deviation_results * get_standard_deviation_at_time_step(results, timestep) for
                            timestep, mean_val in enumerate(mean_results)]
        mean_plus_x_std = [mean_val + self.config.standard_deviation_results * get_standard_deviation_at_time_step(results, timestep) for
                           timestep, mean_val in enumerate(mean_results)]
        return mean_minus_x_std, mean_results, mean_plus_x_std

    def _hide_spines(self, ax, spines_to_hide):
        """Hides splines on a matplotlib image"""
        for spine in spines_to_hide:
            ax.spines[spine].set_visible(False)

    def _save_obj(self, obj, name):
        """Saves given object as a pickle file"""
        if name[-4:] != ".pkl":
            name += ".pkl"
        with open(name, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
