# Contributing

Thanks for contributing to this project. These guidelines keep contributions
clear, reviewable, and reproducible.

## Development setup

1. Use Python 3.11.
2. Create and activate a virtual environment.
3. Install dependencies.

```bash
python -m pip install -r requirements.txt
```

## Workflow

1. Fork the repository and create a feature branch.
2. Keep pull requests focused on one logical change.
3. Include a short description of the change and why it is needed.
4. Reference related issues when relevant.

## Validation before opening a pull request

Run the packaging and validation flow with a local export:

```bash
python scripts/pack_release.py --project_root . --tts_export_dir ./tts_export --out_dir ./dist
python scripts/validate_dataset.py --dataset_dir ./dist/tts_dataset_ready --json_out ./dist/validate_summary.json
```

## Coding standards

- Preserve transcript content unless a change explicitly targets text cleanup.
- Prefer small, deterministic changes in scripts and docs.
- Update docs when behavior changes.
- Do not commit heavy generated assets (`dist/`, `tts_export/`, `*.zip`).

## Commit style

Use clear commit messages in imperative form, for example:

- `add clipping metric to report generator`
- `fix metadata delimiter handling in validator`
