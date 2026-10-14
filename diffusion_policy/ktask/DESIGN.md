# Does RL on K tasks help held-out tasks, and does that change as K grows?

ManiSkill3 port of the robomimic transfer experiments (2026-10-14).

## Task family (ktask/env.py)

32 pick-and-place tasks on the ManiSkill3 Panda tabletop: 8 objects x 4 targets. Each object needs its own
grasp; each target needs its own placement.

| objects (grasp) | targets (placement) |
|---|---|
| cube (centered, 90° symmetric) | goal: on the table within 3 cm of a marked spot |
| bar (across the 2.4 cm side, yaw matters) | platform: on top of a 10 x 10 cm platform, 6 cm high |
| cylinder (any yaw) | bin: inside a 13 x 13 cm bin with 4 cm walls |
| puck (low and wide) | air: held within 2.5 cm of a point 10-22 cm up |
| tall box (grasp high, tips over) | |
| plate (low, across the short side) | |
| tee (across the stem, off-center) | |
| ring (square frame, grasped across one wall) | |

- Randomization: the object pose (xy and yaw) on the right; the target position on the left; air height.
- Observation (40-D, state): tcp pose; gripper width; object pose; object-tcp offset; target point;
  target-object offset; object one-hot (8); target one-hot (4).
- Action: absolute tcp pose (6D rotation) + gripper, run by pd_ee_pose; 20 Hz; 200-step limit.
- Episodes end at the first success (as in robomimic, where they ended at first peg contact).
- Speed: about 17k env-steps/s with 1,024 parallel envs on a 2080 Ti (one robosuite PPO job ran about
  800/s on 32 CPUs).

## Scripted operator and data (ktask/expert.py, gen_data.py)

- The waypoint operator plans from true poses.
- Quality q scales its mistakes: grasp offset, yaw and height; early close; placement offset; release
  height; per-step jitter.
- Success by q: q = 0 → 100% on all 32 tasks; 0.5 → 93%; 1.0 → 59%; 1.5 → 31%.
- Pretraining pool: q ~ U(0, 1.5), 400 episodes per task, every attempt kept (73% successful).
- Expert pool: q = 0, 200 per task.

## Splits (ktask/data.py; Latin design, so every object and target appears in every split)

| split | tasks | role |
|---|---|---|
| U (8) | object o with target o mod 4 | never in pretraining: zero-shot and few-shot learnability |
| H (8) | object o with target o+1 mod 4 | in pretraining, never trained with RL: the fixed held-out set |
| R (16) | the other two targets of each object | in pretraining, eligible for RL: training sets are drawn from these |

Base policy: the robomimic flow-matching transformer (8 layers, 256-d, horizon 16, 2 observations,
8 executed actions, 10 Euler steps). It is trained with BC on the mixed pool of R ∪ H (24 tasks); 3 seeds.

## E1 (main): PPO on K tasks with a fixed total budget

- K ∈ {1, 2, 4, 8, 16}.
- Training sets are nested chains drawn from R and balanced. The K = 2 and K = 4 sets use distinct objects
  and distinct targets. The K = 8 set has every object once and every target twice. K = 16 is all of R.
- Several chains per base seed. The chains give the K-trend a paired comparison along each chain.
- Each iteration runs 1,024 env slots, split evenly over the K tasks. Total episodes are the same for
  every K, so K = 1 gets all of them on one task and K = 16 gets 1/16 per task.
- Checkpoints at 1/16, 1/8, 1/4, 1/2 and all of the budget.
- PPO is DPPO-style as on robosuite (σ on all 10 Euler steps, chain log-probabilities, sparse reward).
  Hyperparameters are re-tuned in a pilot for 1,024 envs.
- Every checkpoint is evaluated on all 32 tasks: 128 episodes per task, with the same start states for
  every checkpoint (paired).

Measured per run and checkpoint (change vs the base, paired bootstrap):

- the K training tasks;
- H (fixed, the primary held-out measure);
- U (zero-shot, never in pretraining);
- the rest of R (in pretraining, not trained);
- each held-out task by its relation to the training set: shares the object, shares the target, both, or
  neither. This is the robomimic first pair (same nut, new peg) and second pair (new nut and peg) generalized.

Comparisons:

1. Held-out change vs K at the same total episodes.
2. Held-out change vs training-task change: (ΔTrain, ΔHeld-out) paths for each K, so that K is compared at
   matched improvement.
3. Held-out change per unit of policy movement (KL from the base on held-out states).

## E2: same data, different update rule (is it RL, or the data?)

For K ∈ {1, 4, 16}, filtered BC is trained on PPO's own logged successful episodes (same episodes, same
start), as on robosuite where it matched PPO on A and broke B. Question: does supervised learning on PPO's
data stop hurting held-out tasks as K grows?

Also SFT on clean expert episodes of the same K tasks.

## E3: learnability of new tasks

From the base and from PPO checkpoints (K = 1, 4, 16), SFT on k clean demos of the U tasks, k ∈ {5, 20}.
Question: does RL on more tasks make new tasks faster to learn?

## Side question: speed

The evaluation records steps to success. So speed transfer (training tasks vs held-out) can be read for
every K. A time-cost arm can be added if PPO alone doesn't change speed.
