# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import argparse
import sys

from maro.utils.exception.cli_exception import CommandNotFoundError


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, examples=None, **kwargs):
        self.examples = examples

        super().__init__(add_help=False, **kwargs)

        self.formatter_class = HelpFormatter

    def format_help(self):
        """Customized format_help

        Returns:
            None
        """
        formatter = self._get_formatter()

        # Add texts, print examples at the end (if any)
        formatter.add_usage(self.usage, self._actions, self._mutually_exclusive_groups)
        formatter.add_text(self.description)
        for action_group in self._action_groups:
            formatter.start_section(action_group.title.capitalize())  # Capitalized
            formatter.add_text(action_group.description)
            formatter.add_arguments(action_group._group_actions)
            formatter.end_section()
        formatter.add_text(self.examples)

        return formatter.format_help()

    def error(self, message):
        """Customized error

        Args:
            message: error message from argparse

        Returns:
            None
        """

        args = sys.argv

        # if get 'help', print help without printing errors
        if '--help' in args or '-h' in args:
            self.print_help()
            # Otherwise it will print traceback here.
            sys.exit(0)
        else:
            # Otherwise, print usage and error messages
            raise CommandNotFoundError(message=message, usage=self.format_usage())


class HelpFormatter(argparse.RawTextHelpFormatter):
    def add_usage(self, usage, actions, groups, prefix=None):
        if prefix is None:
            prefix = 'Usage: '  # Capitalized
        return super(HelpFormatter, self).add_usage(usage, actions, groups, prefix)
