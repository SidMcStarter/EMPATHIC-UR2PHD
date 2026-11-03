# Repository Guidelines

## Project overview

EMPATHIC is the research code for *The EMPATHIC Framework for Task Learning
from Implicit Human Feedback*. Root scripts train and evaluate a neural
reaction mapping from facial Action Units and head pose to three reward
classes. `RobotaxiEnv/` is a Git submodule containing the interactive Robotaxi
environment and online reward-learning agent.

Preserve the paper's algorithms. Compatibility work may change device moves,
imports, checkpoint loading, paths, dependency declarations, and generated
output handling. Do not change model architecture, losses, reward mappings,
training logic, game logic, or online belief updates unless explicitly asked.

## Storage and privacy rules

- This project can consume substantial disk space through the dataset,
  checkpoints, OpenFace source/build files, Homebrew dependencies, webcam
  frames, and generated CSV files. Before downloading or building large
  dependencies, inspect available space with `df -h` and relevant directories
  with `du -sh <path>` and tell the user the expected action.
- Reuse the single root `venv/`; do not create environments inside the
  Robotaxi submodule.
- Do not install OpenFace, download datasets, or perform other large downloads
  unless the user explicitly requests it.
- Never delete datasets, checkpoints, participant data, environments, or build
  directories without explicit approval.
- Do not commit datasets, checkpoints, camera frames, OpenFace output, logs,
  CSV output, screenshots, videos, audio logs, caches, or virtual environments.
- The GUI and online-learning workflow may access the webcam. Do not launch
  either one or collect participant data without explicit user approval.

## Clone and Python environment

Clone with Robotaxi included:

```bash
git clone --recursive <repository-url>
cd EMPATHIC
```

For an existing clone whose submodule is absent:

```bash
git submodule update --init --recursive
```

Use one virtual environment at the repository root:

```bash
python3.10 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

The repository was written for an older Python ecosystem. On modern Python,
inspect installation failures instead of broadly upgrading code or changing
the algorithm. Known packaging issues may include the old Pillow pin and the
deprecated `sklearn` package name; prefer compatible Pillow and
`scikit-learn` packages while preserving runtime behavior.

Run training and evaluation scripts from the repository root. Run ordinary
Robotaxi commands from `RobotaxiEnv/`, where its relative asset paths resolve.
Run `online_learning.py` from the repository root.

## Dataset layout

Extract the preprocessed dataset into `detected/`. It should contain shared
episode statistics and one directory per participant, approximately:

```text
detected/
  RLstatistics_episode_<episode>.csv
  <subject-id>/
    <subject-id>_episode_<episode>_detectionFrames_0.csv
    <subject-id>_episode_<episode>_detectionFrames_1.csv
    <subject-id>_episode_<episode>_detectionFrames_2.csv
    <subject-id>_episode_<episode>_detectionFrames_holdout.csv
```

The numbered files represent data splits, not three camera frames at one time
step. Holdout files are used for final subject evaluation. Do not move or
rename dataset files unless a loader requires a documented correction.

## Training and offline evaluation

The root training and test scripts already select MPS, CUDA, or CPU. Preserve
that device behavior and keep annotation tensors in `float32` for MPS.

Create output directories if absent:

```bash
mkdir -p MLP_facs_reward_models evaluation_outputs
```

Train a per-subject model:

```bash
./venv/bin/python train_mlp_net_facs.py <subject-id>
```

Train the shared model over all subjects:

```bash
./venv/bin/python train_mlp_net_facsall.py
```

Evaluate the shared model on a subject's holdout data:

```bash
./venv/bin/python test_facs.py <subject-id>
```

`test_all_subjects.ipynb` contains one test cell per subject. Keep notebook
outputs visible for local analysis, but do not commit executed output unless
explicitly requested. Evaluation reports the best reward ordering and
Kendall's tau against the correct `(0, 1, 2)` ordering.

## Robotaxi submodule

`RobotaxiEnv` points to
`https://github.com/Pearl-UTexas/RobotaxiEnv.git` at commit `59fe65e`. The
parent repository stores only the submodule commit. Local submodule edits are
not preserved by a parent commit and must be reapplied after a fresh clone
while this project continues to use the upstream repository directly.

Apply these narrow compatibility fixes when needed:

- In `robotaxi/detectors/feedback_detector.py`, use
  `from .network_facs import Net`.
- In `play.py`, import only classes exported by `robotaxi.agent` for the
  supported `human`, `random`, `val-itr`, `mixed`, and `reward-learning`
  paths. Do not claim legacy choices work without restoring and testing them.
- In the nested feedback detector and
  `robotaxi/detectors/network_facs.py`, select MPS, CUDA, or CPU; replace
  `.cuda()` with `.to(device)`; create constants on the input device; remove
  unnecessary `Variable` wrappers; and load checkpoints with
  `map_location=device`.
- Add `/user_study_data/` to `RobotaxiEnv/.gitignore`.

Safe non-camera smoke test, run from `RobotaxiEnv/`:

```bash
../venv/bin/python play.py --agent mixed --interface cli --num-episodes 1 --test_run
```

GUI test, only with explicit camera approval:

```bash
../venv/bin/python play.py --agent mixed --interface gui --num-episodes 1 --test_run
```

The `mixed` agent randomly switches between predefined value-iteration
policies and does not use the trained facial model.

## OpenFace and online reward learning

OpenFace is a separate native C++ application, not a Python package. It reads
the webcam and writes facial Action Units and head pose to
`openface_out/<participant-id>/webcam.csv`. The feedback detector tails that
CSV and sends a rolling 23-frame feature window to the reaction model.

For a storage-conscious macOS installation, clone OpenFace shallowly outside
this repository and build Release only:

```bash
brew install boost tbb openblas opencv dlib
cd <external-source-directory>
git clone --depth 1 https://github.com/TadasBaltrusaitis/OpenFace.git
cd OpenFace
bash download_models.sh
cmake -S . -B build -D CMAKE_BUILD_TYPE=RELEASE
cmake --build build -j4
```

Do not run those installation commands automatically. First confirm free disk
space and obtain user approval. On Apple Silicon, expect possible build fixes
because the upstream macOS instructions are old. Prefer Apple Clang first;
install GCC or XQuartz only if the build demonstrates they are needed.

After building, update `start_openface.bash` so its placeholder points to the
absolute `OpenFace/build/bin/FeatureExtraction` binary. Test OpenFace directly
before involving Robotaxi:

```bash
<openface-path>/build/bin/FeatureExtraction \
  -device 0 \
  -out_dir /tmp/openface_test \
  -of webcam.csv
```

Confirm that the generated CSV header still matches the fixed feature indexes
used by `feedback_detector.py`; do not assume every OpenFace version has an
identical column layout.

The online workflow loads
`online_learning_reward_models/facsposeauxall_1.0458.pkl`. It is a reaction
classifier, not the driving policy. Do not replace it with
`MLP_facs_reward_models/allsubjects_1.0439.pkl` unless architecture, feature
layout, and 23-frame window compatibility have been verified.

The intended online entry point, run from the repository root, is:

```bash
./venv/bin/python online_learning.py \
  --agent reward-learning \
  --participant <participant-id> \
  --num-episodes 1 \
  --test_run
```

This starts OpenFace, the Robotaxi GUI, the trained reaction model, and the
belief-based `RewardLearningAgent`. The agent begins with a belief over all six
rankings of passenger, roadblock, and parked-car rewards. Facial-model output
updates that belief after encounters, and the belief selects among
value-iteration policies.

## Verification and Git boundaries

- Keep verification proportional. Prefer syntax checks and CLI smoke tests
  before full training or camera runs.
- After device edits, search touched code for `.cuda(`, `.cuda()`,
  `torch.cuda`, and explicit CUDA device strings. Report unrelated matches.
- Check parent and submodule status separately:
  `git status --short` and `git -C RobotaxiEnv status --short`.
- Never commit a parent submodule pointer to a local-only Robotaxi commit; a
  future clone could not fetch it from upstream.
- Do not commit executed notebook output, generated artifacts, participant
  data, or unrelated working-tree changes.
