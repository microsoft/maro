# Contributing to MARO

MARO is newborn for Reinforcement learning as a Service (RaaS) in the resource optimization domain. Your contribution is precious to make RaaS come true.

- [Open issues](https://github.com/microsoft/maro/issues) for reporting bugs and requesting new features.
- Contribute to [examples](https://github.com/microsoft/maro/tree/master/examples) to share your problem modeling to others.
- Contribute to [scenarios](https://github.com/microsoft/maro/tree/master/maro/simulator/scenarios) to provide more meaningful environments.
- Contribute to [topologies](https://github.com/microsoft/maro/tree/master/maro/simulator/scenarios/citi_bike/topologies) to enhance existing MARO scenarios.
- Contribute to [algorithms](https://github.com/microsoft/maro/tree/master/maro/rl/algorithms) to enrich MARO RL libraries.
- Contribute to [orchestration](https://github.com/microsoft/maro/tree/master/maro/cli) to broad MARO supported cloud services.
- Contribute to [communication](https://github.com/microsoft/maro/tree/master/maro/communication) to enhance MARO distributed training capacity.
- Contribute to [tests](https://github.com/microsoft/maro/tree/master/tests) to make it more reliable and stable.
- Contribute to [documentation](https://github.com/microsoft/maro/tree/master/maro) to make it straightforward for everyone.

## Notes

- Check Style

  Please make sure lint your code, and pass the code checking before pull request.

  We have prepared a configuration file for isort and flake8 to lint.

  ```sh
  # Install isort.
  pip install isort

  # Automatically re-format your imports with isort.
  isort --settings-path .github/linters/tox.ini

  # Install flake8.
  pip install flake8

  # Lint with flake8.
  flake8 --config .github/linters/tox.ini

  ```

- [Update Change Log](https://github.com/github-changelog-generator/github-changelog-generator#installation) (if needed)

  ```sh
  # Use --token, when accessing limitation happens.
  # -t, --token [TOKEN]              To make more than 50 requests per hour your GitHub token is required. You can generate it at: https://github.com/settings/tokens/new
  github_changelog_generator -u microsoft -p maro --max-issues 5
  ```

- Enable [EditorConfig](https://editorconfig.org/#download) in your editor.
