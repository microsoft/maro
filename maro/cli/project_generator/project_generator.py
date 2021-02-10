# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pprint

import stringcase
from jinja2 import Environment
from jinja2.loaders import PackageLoader
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.validation import ValidationError, Validator

import maro
from maro.simulator.utils.common import get_scenarios, get_topologies

# Default topology to use to generate customize topology.
default_topologies = {
    "cim": "toy.4p_ssdd_l0.0",
    "citi_bike": "toy.3s_4t",
    "vm_scheduling": "azure.2019.10k"
}


class NonEmptyStringValidator(Validator):
    """Validator used to validate to make sure input is not empty string."""
    def __init__(self, err_msg: str):
        super().__init__()
        self._err_msg = err_msg

    def validate(self, document):
        if not document.text or document.text.strip() == "":
            raise ValidationError(message=self._err_msg)


class PositiveNumberValidator(Validator):
    """Validator used to make sure input value is a positive number."""
    def validate(self, document):
        text = document.text

        if not text or not text.isdigit():
            raise ValidationError(message="Only positive numbers are accepted.")


def is_command_yes(cmd: str):
    """Check if user agree with current command.

    Args:
        cmd (str): Command input value.

    Returns:
        bool: True if command value is match "yes" or "y" (case insensitive.).
    """
    lower_cmd = cmd.lower()

    return lower_cmd == "yes" or lower_cmd == "y"


def new_project(**kwargs: dict):
    """Create a new project."""
    use_builtin_scenario = is_command_yes(prompt("Use built-in scenario?", default="yes"))

    if use_builtin_scenario:
        generate_function = new_project_with_builtin_scenario

        # Build built-in scenario completer.
        builtin_scenarios = get_scenarios()

        # Validator for scenario name to make sure input is a built-in scenario.
        class BuiltInScenarioValidator(Validator):
            """Validate is input scenario is built-in one."""
            def validate(self, document):
                if document.text not in builtin_scenarios:
                    raise ValidationError(message="Scenario name not a built-in one.")

        builtin_scenario_completer = WordCompleter(builtin_scenarios)

        # Select a scenario.
        scenario_name = prompt(
            "Scenario name:",
            default=builtin_scenarios[0],
            completer=builtin_scenario_completer,
            complete_while_typing=True,
            validate_while_typing=True,
            validator=BuiltInScenarioValidator()
        )

        use_builtin_topology = is_command_yes(prompt(
            "Use built-in topology (configuration)?",
            default="yes"
        ))

        if use_builtin_topology:
            # Build topology completer for topology selecting.
            builtin_topologies = get_topologies(scenario_name)
            builtin_topologies_completer = WordCompleter(builtin_topologies)

            class BuiltinTopologyValidator(Validator):
                """Validate if input topology is built-in one."""
                def validate(self, document):
                    if document.text not in builtin_topologies:
                        raise ValidationError(message="Topology not exist.")

            topology_name = prompt(
                "Topology name to use:",
                default=builtin_topologies[0],
                completer=builtin_topologies_completer,
                validator=BuiltinTopologyValidator(),
                validate_while_typing=True,
                complete_while_typing=True
            )
        else:
            topology_name = prompt(
                "Topology name to create (content is copied from built-in topology.):",
                validator=NonEmptyStringValidator("Topology name cannot be empty."),
                validate_while_typing=True
            )
    else:
        use_builtin_topology = False
        generate_function = new_project_from_scratch
        scenario_name = os.path.basename(os.getcwd())

        scenario_name = prompt(
            "New scenario name:",
            default=scenario_name,
            validator=NonEmptyStringValidator("Scenario name cannot be empty.")
        )

        topology_name = prompt(
            "New topology name:",
            default=scenario_name,
            validator=NonEmptyStringValidator("Topology name cannot be empty.")
        )

    # Misc settings.
    durations = prompt(
        "Durations to emulate:",
        default="100",
        validator=PositiveNumberValidator(),
        validate_while_typing=True
    )

    episodes = prompt(
        "Number of episodes to emulate:",
        default="10",
        validator=PositiveNumberValidator(),
        validate_while_typing=True
    )

    options = \
        {
            "use_builtin_scenario": use_builtin_scenario,
            "scenario": scenario_name,
            "total_episodes": int(episodes),
            "durations": int(durations),
            "use_builtin_topology": use_builtin_topology,
            "topology": topology_name
        }

    should_continue = prompt(
        pprint.pformat(options) + "\n\nIs this OK?",
        default="yes"
    )

    if is_command_yes(should_continue):
        generate_function(options)


def new_project_with_builtin_scenario(options: dict):
    """Create a runner.py to run with built-in scenario.

    Args:
        options (dict): Options to generate runner.py.
    """
    if not options["use_builtin_topology"]:
        # Create a folder to place new topology.
        topology_root_path = os.path.join(os.getcwd(), "topologies")

        # Make topology root and itself folder exist.
        if not os.path.exists(topology_root_path):
            os.mkdir(topology_root_path)

        topology_path = os.path.join(topology_root_path, options["topology"])

        if not os.path.exists(topology_path):
            os.mkdir(topology_path)

        # Copy content from exist one.
        scenario_name = options["scenario"]
        topology_to_copy = default_topologies.get(scenario_name, None)

        if topology_to_copy:
            topology_to_copy_path = os.path.join(
                maro.__path__[0],
                "simulator",
                "scenarios",
                scenario_name,
                "topologies",
                topology_to_copy
            )

        # If the default one not exist, then use first one.
        if not os.path.exists(topology_to_copy_path):
            topology_to_copy = get_topologies(options["scenario"])[0]
            topology_to_copy_path = os.path.join(
                maro.__path__[0],
                "simulator",
                "scenarios",
                scenario_name,
                "topologies",
                topology_to_copy
            )

        with open(os.path.join(topology_to_copy_path, "config.yml"), "rt") as source_fp:
            with open(os.path.join(topology_path, "config.yml"), "wt") as dest_fp:
                dest_fp.write(source_fp.read())

        options["topology"] = f"topologies/{options['topology']}"

    env = generate_environment()

    generate_runner(env, options)


def new_project_from_scratch(options: dict):
    """Create a project to customize scenario.

    Args:
        options (dict): Options for new project.
    """
    # Scenario folder
    scenario_folder = os.path.join(os.getcwd(), "scenario")

    if not os.path.exists(scenario_folder):
        os.mkdir(scenario_folder)

    topology_root = os.path.join(scenario_folder, "topologies")

    if not os.path.exists(topology_root):
        os.mkdir(topology_root)

    scenario_name = options["scenario"]
    topology_name = options["topology"]

    options["scenario_cls"] = stringcase.pascalcase(scenario_name)
    options["topology"] = f"scenario/topologies/{options['topology']}"

    topology_folder = os.path.join(topology_root, topology_name)

    if not os.path.exists(topology_folder):
        os.mkdir(topology_folder)

    env = generate_environment()

    # Generate runner.py.
    generate_runner(env, options)

    # Generate scenario/business_engine.py.
    generate_business_engine(env, scenario_folder, options)

    # Generate scenario/frame_build.py.
    generate_frame_builder(env, scenario_folder, options)

    # Generate scenario/common.py that contains dummy DecisionEvent and Action class.
    generate_commons(env, scenario_folder, options)

    # Generate scenario/events.py that contains dummy events definitions.
    generate_events(env, scenario_folder, options)

    # Generate a empty config.yml.
    with open(os.path.join(topology_folder, "config.yml"), "wt") as fp:
        fp.write("# write your own configuration here, or use other format.")


def generate_environment():
    """Generate a common template environment."""
    env = Environment(
        loader=PackageLoader("maro", "cli/project_generator/templates"),
        trim_blocks=True
    )

    return env


def generate_runner(env: Environment, options: dict):
    """Generate runner.py with options.

    Args:
        env (Environment): Template environment used to retrieve template.
        options (dict): Options for runner.py.
    """
    use_builtin_scenario = options.get("use_builtin_scenario", False)

    if use_builtin_scenario:
        template_name = "runner.base.py.jinja"
    else:
        template_name = "runner.customize.py.jinja"

    template = env.get_template(template_name)

    with open(os.path.join(os.getcwd(), "runner.py"), "wt") as fp:
        fp.write(template.render(project=options))


def generate_business_engine(env: Environment, folder: str, options: dict):
    """Generate business_engine.py with options.

    Args:
        env (Environment): Template environment used to retrieve template.
        options (dict): Options for business_engine.py.
    """
    template = env.get_template("business_engine.py.jinja")

    with open(os.path.join(folder, "business_engine.py"), "wt") as fp:
        fp.write(template.render(project=options))


def generate_frame_builder(env: Environment, folder: str, options: dict):
    """Generate frame_builder.py with options.

    Args:
        env (Environment): Template environment used to retrieve template.
        folder (str): Folder to place new file.
        options (dict): Options for frame_builder.py.
    """
    template = env.get_template("frame_builder.py.jinja")

    with open(os.path.join(folder, "frame_builder.py"), "wt") as fp:
        fp.write(template.render(project=options))


def generate_commons(env: Environment, folder: str, options: dict):
    """Generate common.py with options.

    Args:
        env (Environment): Template environment used to retrieve template.
        folder (str): Folder to place new file.
        options (dict): Options for common.py.
    """
    template = env.get_template("common.py.jinja")

    with open(os.path.join(folder, "common.py"), "wt") as fp:
        fp.write(template.render(project=options))


def generate_events(env: Environment, folder: str, options: dict):
    """Generate events.py with options.

    Args:
        env (Environment): Template environment used to retrieve template.
        folder (str): Folder to place new file.
        options (dict): Options for events.py.
    """
    template = env.get_template("events.py.jinja")

    with open(os.path.join(folder, "events.py"), "wt") as fp:
        fp.write(template.render(project=options))
