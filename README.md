# RLIF on RLPD

Simulation-only research code for Reinforcement Learning via Intervention Feedback (RLIF) on top of an RLPD/SAC-style off-policy learner, implemented in PyTorch for Gymnasium/MuJoCo continuous-control tasks.

## What is included

- Behavior-cloning expert training from offline demonstrations.
- Reference Q training for value-based interventions.
- RLIF rollout collection with random or value-based intervention models.
- RLPD-style online learning with SAC actor-critic updates, LayerNorm critics, critic ensembles, and mixed offline/online replay.
- Round-based RLIF training with intervention labels, checkpointed resume support, and D4RL-style normalized return evaluation.
- Gymnasium MuJoCo task support for Hopper, Walker2d, and configurable Adroit Pen variants.

## Installation

```bash
python -m pip install -e .
```

For Adroit Pen, install a Gymnasium-compatible robotics package that exposes `AdroitHandPen-v1`, and provide an offline demo dataset in the expected `.npz` format.

## Dataset format

The default loaders expect an `.npz` file with at least:

- `observations`
- `actions`
- `next_observations`
- `rewards`
- `terminals`

If you have a D4RL-style environment with `get_dataset()`, the code can also pull the dataset directly.
The runtime path in this repository is Gymnasium-only for simulation; offline data should be provided as a local `.npz` file.

## How to get the dataset

This code does not ship expert demonstrations. You have two practical options:

1. Download or export a standard offline RL dataset and convert it to `.npz` with the required keys.
2. Record your own trajectories in Gymnasium and save them as `.npz` using the same field names.

For Hopper and Walker2d, use a standard offline RL dataset from your preferred source and cache it locally as `data/hopper_expert.npz` or `data/walker2d_expert.npz`. RLIF warm-starts from the first 50 Hopper trajectories or the first 10 Walker2d trajectories by default. For Adroit Pen, do the same with an expert demonstration bundle for the hand task.

If you do not have a local dataset yet, set `data.offline_path` and `data.expert_demo_path` to the file you created after downloading or exporting the offline trajectories.

The `python train_expert.py --config configs/hopper.yaml` command assumes that `data/hopper_expert.npz` already exists. If it does not, create or download the dataset first, then rerun the command.

### Export from Minari

If you want to use the Minari-hosted expert datasets, export them once and save them into the repo format:

```bash
python -m pip install minari
python scripts/export_minari_to_npz.py --dataset-id D4RL/hopper/expert-v0 --output data/hopper_expert.npz
python scripts/export_minari_to_npz.py --dataset-id D4RL/walker2d/expert-v0 --output data/walker2d_expert.npz
```

After that, run training normally:

```bash
python train_expert.py --config configs/hopper.yaml
python train_rlif.py --config configs/hopper.yaml
```

## Train the expert

```bash
python train_expert.py --config configs/hopper.yaml
```

With W&B online logging and periodic eval-return tracking:

```bash
python train_expert.py --config configs/hopper.yaml --wandb-project RLIF-Hopper-Expert --wandb-run-name hopper-expert-seed1 --wandb-mode online --eval-interval-epochs 5 --eval-episodes 5
```

This trains:

- a behavior-cloning policy for intervention actions
- a reference Q model used by value-based interventions

Outputs are written to `outputs/<env>/expert/`.

## Run RLIF

```bash
python train_rlif.py --config configs/hopper.yaml
```

To log RLIF runs to Weights & Biases, add `--wandb-project` and optionally `--wandb-entity`, `--wandb-run-name`, and `--wandb-mode`.

For Hopper, a good default W&B setup is:

```bash
python train_rlif.py --config configs/hopper.yaml --wandb-project RLIF-Hopper --wandb-run-name hopper-value-seed0 --wandb-mode online
```

Recommended names:

- Project: `RLIF-Hopper`
- Entity: your W&B account or team name
- Run name: `hopper-value-seed0`

If online logging returns a permission error, rerun with `--wandb-mode offline` and sync later, or change `--wandb-entity` / `--wandb-project` to ones you own.

This will:

1. Load the offline trajectories into the replay buffer.
2. Load the trained expert policy and reference Q.
3. Pretrain the learner for 200 epochs with 300 gradient steps per epoch.
4. Run 100 RLIF rounds, collecting 5 trajectories per round.
5. Label the transition immediately preceding each intervention with reward `-1` and set all other collected rewards to `0`.
6. Train the RLPD learner with mixed offline/online batches and the task-specific UTD ratio.

The learner saves `latest.pt` and `checkpoint_round_*.pt` files under `outputs/<env>/rlif/`. To resume, pass `--resume-from outputs/<env>/rlif/latest.pt`.

Example runs:

```bash
python train_rlif.py --config configs/hopper.yaml --wandb-project RLIF-Hopper --wandb-run-name hopper-value-seed0 --wandb-mode online
python train_rlif.py --config configs/walker2d.yaml --wandb-project RLIF-Walker2d --wandb-run-name walker2d-value-seed0 --wandb-mode online
```

Checkpoints are saved under `outputs/<env>/rlif/`.

## Evaluate checkpoints

Evaluate a learner checkpoint:

```bash
python evaluate.py --config configs/hopper.yaml --checkpoint outputs/hopper/rlif/final.pt --mode learner
```

Evaluate an expert checkpoint:

```bash
python evaluate.py --config configs/hopper.yaml --checkpoint outputs/hopper/expert/expert_actor.pt --mode expert
```

## Change environments

Switch the `--config` argument to one of:

- `configs/hopper.yaml`
- `configs/walker2d.yaml`
- `configs/adroit_pen.yaml`

To add a new task, copy one of the YAML files and change:

- `env.env_id`
- `env.max_episode_steps`
- dataset paths under `data`
- `data.offline_trajectory_limit`
- any environment-specific `rlpd` or `rlif` settings

## Notes on the method

- RLIF differs from DAgger because the online policy is not trained to imitate expert actions directly at every intervention point. Instead, the expert provides intervention control, and the learner receives a sparse negative reward signal on the transition immediately before the intervention.
- The expert is trained by behavioral cloning on offline demonstrations by default. A reference Q model is trained on the same expert data so the intervention oracle can use value comparisons.
- Intervention rewards are created by retroactively changing the reward of the transition immediately preceding an intervention to `-1`; all other online transitions receive `0`.
- RLPD is adapted by sampling the offline and online replay buffers directly in each learner batch, using a critic ensemble with LayerNorm, and keeping SAC-style actor-critic updates with optional entropy backups.
- Evaluation reports normalized return rather than success rate for Hopper and Walker2d.
