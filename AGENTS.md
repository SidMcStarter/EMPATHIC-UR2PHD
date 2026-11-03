# Repository Guidelines

## Project overview

EMPATHIC is the research code for *The EMPATHIC Framework for Task Learning
from Implicit Human Feedback*. The root Python scripts train and evaluate the
facial-action and pose reward classifier. `RobotaxiEnv/` is a Git submodule
containing the interactive Robotaxi user-study environment.

Preserve the paper's algorithms. Compatibility work may change device moves,
imports, checkpoint loading, paths, and generated-output handling, but must not
change model architecture, losses, reward mappings, training logic, game logic,
or online reward-learning behavior unless the user explicitly requests it.

## Clone and setup

- Clone with submodules:
  `git clone --recursive <repository-url>`
- Initialize Robotaxi in an existing clone:
  `git submodule update --init --recursive`
- Create and activate a virtual environment at the repository root, then
  install the root requirements. The original dependency pins are old, so
  inspect compatibility before changing package versions.
- Run root scripts from the repository root. Run Robotaxi commands from
  `RobotaxiEnv/` so its relative asset and output paths resolve correctly.

## Robotaxi submodule

`RobotaxiEnv` points to the public upstream repository
`https://github.com/Pearl-UTexas/RobotaxiEnv.git` at commit `59fe65e`. The parent
repository records only the submodule commit. Edits made inside
`RobotaxiEnv/` are not included in a parent-repository commit and must be
reapplied after a fresh clone while the project continues to use upstream
directly.

For current macOS, CPU, and CUDA compatibility, inspect and apply these narrow
fixes when needed:

- In `robotaxi/detectors/feedback_detector.py`, import the model with
  `from .network_facs import Net`.
- In `play.py`, import only agent classes exported by `robotaxi.agent` for the
  supported `human`, `random`, `val-itr`, `mixed`, and `reward-learning` paths.
  Do not claim unsupported legacy agent choices work without restoring their
  implementations and testing them.
- In the nested feedback detector and `robotaxi/detectors/network_facs.py`, use
  one MPS/CUDA/CPU device selector, replace hard-coded `.cuda()` moves with
  `.to(device)`, create constants on the input tensor's device, and load
  checkpoints with `map_location=device`.
- Add `/user_study_data/` to `RobotaxiEnv/.gitignore`; camera frames and
  participant data must remain untracked.

## Generated and private data

Do not commit datasets, model checkpoints, participant camera frames, OpenFace
output, logs, CSV output, screenshots, videos, audio logs, caches, or virtual
environments. Preserve any existing local data and never delete generated data
unless the user explicitly requests it.

The Robotaxi GUI starts webcam capture automatically. Do not launch the GUI,
access the webcam, or collect participant data without explicit user approval.
Use CLI mode for non-camera smoke tests.

## Useful commands

- Root subject training:
  `./venv/bin/python train_mlp_net_facs.py <subject-id>`
- Shared-model training:
  `./venv/bin/python train_mlp_net_facsall.py`
- Subject reward-ranking evaluation:
  `./venv/bin/python test_facs.py <subject-id>`
- Robotaxi CLI smoke test, run from `RobotaxiEnv/`:
  `../venv/bin/python play.py --agent mixed --interface cli --num-episodes 1 --test_run`
- Robotaxi GUI with camera, only when explicitly requested:
  `../venv/bin/python play.py --agent mixed --interface gui --num-episodes 1 --test_run`

## Verification and Git boundaries

- Keep verification proportional to the change. Use syntax checks and focused
  CLI tests before attempting full training or camera-based runs.
- Search touched files for remaining hard-coded CUDA operations after device
  compatibility edits and report unrelated occurrences rather than broadening
  the change silently.
- Check parent and submodule status separately with `git status --short` and
  `git -C RobotaxiEnv status --short`.
- Do not commit executed notebook output, generated artifacts, or unrelated
  working-tree changes.
