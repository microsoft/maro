![MARO LOGO](./docs/source/images/logo.svg)
MARO(Multi-Agent Resource Optimization) is a multi-agent resource optimization platform, which is an end-to-end solution for both academic research and industry application. A super-fast and highly extensible simulator system and a scalable distributed system are provided, which can well support both small scale single host exploration and large scale distributed application. Some out-of-box scenarios, algorithms and related baselines are provided for a quick hands-on exploration.

## Contents

| File/folder       | Description                                |
|-------------------|--------------------------------------------|
| `maro`            | MARO source code.                          |
| `examples`        | Showcase of MARO.                          |
| `tools`           | Gulp-based helper scripts.                 |
| `notebooks`       | MARO quick-start notebooks.                |

## Run from Source Code
### Rerequisites
- [Python >= 3.6](https://www.python.org/downloads/)
- C++ Compiler
    - Linux or Mac OS X: `gcc`
    - Windows: [Build Tools for Visual Studio 2017](https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=15)

### Build MARO

```sh
pip install -r requirements.dev.txt
export PYTHONPATH=$PWD:$PYTHONPATH
python maro/utils/dashboard/package_data.py
bash build_maro.sh
```

### Run Examples

```sh
cd examples/ecr/q_learning/single_host_mode
bash silent_run.sh
```

### Run in Docker `(only support linux by now)`
Refer to [prerequisites](./tools/README.md) for details.

```sh
cd tools
gulp l/build_image
gulp l/launch_container
gulp l/login_container
cd examples/ecr/q_learning/single_host_mode
bash silent_run.sh
```

### Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Reference Papers `(TODO)`
<!-- TODO -->

## License

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the [MIT](./LICENSE) License.
