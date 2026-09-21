# Contributing

1. Open an issue describing the scientific or engineering change.
2. Work on a branch and keep commits focused.
3. Do not commit non-release intermediate data, credentials, local paths, or generated result trees.
4. Run `python -m pytest -q` and `python -m compileall -q src scripts` before opening a pull request.
5. For changes affecting results, include the resolved configuration, input checksum, seed, software versions, and a short validation note.

Scientific changes must preserve the seed-level export contract and document any change in metrics, preprocessing, folds, or model settings.
