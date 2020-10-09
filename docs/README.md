# MARO Documentation

## Pre-install

```sh
pip install -U -r requirements.docs.txt
```

## Build docs

```sh
# For linux, darwin
make html

# For windows
./make.bat html
```

## Local host

```sh
python -m http.server -d ./_build/html 8000 -b 0.0.0.0
```

## Auto-build/Auto-refresh

### Prerequisites

- [Watchdog](https://pypi.org/project/watchdog/)
- [Browser-sync](https://www.browsersync.io/)

```sh
# Watch file change, auto-build
watchmedo shell-command --patterns="*.rst;*.md;*.py;*.png;*.ico;*.svg" --ignore-pattern="_build/*" --recursive --command="APIDOC_GEN=False make html"
# Watch file change, auto-refresh
browser-sync start --server --startPath ./_build/html --port 8000 --files "**/*"
```
