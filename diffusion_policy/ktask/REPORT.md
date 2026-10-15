# Does RL on K tasks help held-out tasks, and does that change with K? (ManiSkill3, 2026-10-14)

Interim: 37 of 54 E1 runs, E2 29 of 30, E3 11 of 12. The remaining E1 replicates are running; the numbers
below will tighten but not change direction.

## Short answer

1. **Yes, RL helps held-out tasks at every K.**
   - PPO on K training tasks raised success on the fixed held-out set H by +0.07 (K = 1) to +0.12
     (K = 4). H tasks were in pretraining but never trained with RL.
   - It raised success on never-seen object–target pairs by +0.07 to +0.12.
   - Every interval excludes zero.
2. **What changes with K is how much of the gain transfers, and how reliably.** The table below is held-out
   change per unit of training-task change (each run's slope over its 4k–32k checkpoints).

   | K | runs | H per unit | U per unit | runs above 0.5 |
   |---|---|---|---|---|
   | 1 | 9 | 0.69 [0.42, 0.99] | 0.56 [0.36, 0.78] | 6/9 |
   | 2 | 9 | 0.53 [0.42, 0.66] | 0.58 [0.40, 0.75] | 4/9 |
   | 4 | 7 | 0.87 [0.68, 1.08] | 0.82 [0.65, 0.98] | 7/7 |
   | 8 | 7 | 0.95 [0.76, 1.17] | 0.90 [0.70, 1.12] | 7/7 |
   | 16 | 5 | 0.90 [0.69, 1.10] | 0.92 [0.86, 0.99] | 5/5 |

   - From K = 4 on, about 0.9 of the training gain shows up on held-out tasks, in every run.
   - With 1–2 tasks it is about 0.5–0.7 on average and erratic. One single-task run (puck on the platform,
     training +0.47) transferred as broadly as K = 16; others hardly moved held-out tasks.
   - The total budget is fixed, so the gain on the training tasks shrinks a little as K grows (+0.15 →
     +0.11). The absolute held-out gain therefore peaks around K = 4 and stays flat up to 16.
3. **It is the update rule that matters, mainly when K is small** (E2, same tasks, both arms from the
   base).

   | | PPO (training, held out) | SFT on clean expert demos | filtered BC on PPO's own successes |
   |---|---|---|---|
   | K = 1 | +0.16, **+0.07** | +0.15, **−0.09** | +0.00, **−0.21** |
   | K = 4 | +0.13, +0.11 | +0.15, +0.10 | −0.03, −0.07 |
   | K = 16 | +0.13, +0.11 | +0.13, +0.13 | −0.04, −0.06 |

   - On a single task, supervised fine-tuning makes the policy forget the other tasks, as on robosuite.
     This happens even on the same episodes PPO collected. PPO instead improves them.
   - With 4–16 tasks, SFT on expert demos transfers as well as PPO.
   - Filtered BC on PPO's episodes hurts at every K, including the training tasks. I don't have an
     explanation yet.
4. **RL does not make new tasks easier to learn** (E3).
   - After SFT on 5 or 20 clean demos of each never-seen task, every start ends at 0.69–0.71: the base, and
     PPO on 1, 4 or 16 tasks. Pairwise differences are ≤ 0.01.
   - The PPO starts' zero-shot head start (0.58–0.61 vs 0.50) disappears.
5. **Speed does not transfer, because it does not change in the first place.** Steps to success moved by
   at most 3% on any group; sparse-reward PPO here makes the policy succeed more, not faster.

## What transfers

- **Shared skills.** Object-specific and target-specific gains recombine in held-out pairs.
  - In the K = 16 pilot (seed 1), every puck task improved by about +0.4, including the held-out one
    (0.16 → 0.58) and the never-seen one (0.23 → 0.58).
  - The never-seen cube-on-goal went 0.16 → 0.66. Its grasp and its placement were each trained only in
    other pairs.
  - The tee and the tall box barely moved, trained or not.
- **Relation to the training set** (K = 1–2). Held-out tasks sharing the object gained +0.11–0.13, sharing
  the target +0.07–0.09, sharing neither +0.06. Part of the transfer is generic.
- **Both phases of the task.** On held-out tasks, the lift rate rose by +0.04–0.07 and success after a lift
  by +0.05–0.11.
- **Base-specific weaknesses.** The weakest base (seed 1, 0.41 on R) gained the most. For that base, PPO on
  one task that exposed its weakness fixed it everywhere. This suggests more training tasks make transfer
  reliable because some task in the set is likely to expose what is broadly wrong. That is consistent with
  the run-to-run spread at K ≤ 2, but not yet tested directly.

## Caveats

- One task family. Every task's demos come from the same scripted operator with the same noise model, so
  the base's failure modes are partly shared by construction, which favors broad transfer. On robosuite,
  where the round nut needed a different grasp, PPO on A did not help B.
- The total episode budget is fixed across K, so per-task budgets differ (32k episodes for K = 1, 2k per task
  for K = 16).
- K = 16 always trains the same 16 tasks, so its replicates come only from base seeds and PPO seeds.
- The K = 1 training-task changes use 512-episode evaluations (E1x) where available. With 128 episodes, a
  single task's estimate has a standard error of about 0.05.
- The PPO setting came from pilots (critic warmup; lr 3e-6; 1,024 episodes per iteration). Larger
  per-iteration steps first collapsed the policy, then recovered.

## Setup

See ktask/DESIGN.md and the methods notes. In brief:

- **Tasks:** 32 tasks = 8 objects × 4 targets, split into U (8 never pretrained), H (8 held out from RL) and
  R (16 eligible for RL), in a Latin design.
- **Base:** the robosuite flow-matching transformer, trained with BC on mixed-quality scripted demos of
  R ∪ H; 3 seeds.
- **E1:** PPO on nested, balanced chains of K ∈ {1, 2, 4, 8, 16} tasks from R, 32,768 episodes each.
  4 chains × 3 seeds for K < 16; 2 PPO seeds × 3 base seeds for K = 16.
- **Evaluation:** every checkpoint on all 32 tasks, 128 episodes each, with the same start states as the
  base (paired; 95% CIs over runs and start states).
- **E2:** filtered BC on PPO's logged successful episodes and SFT on clean expert demos of the same tasks,
  from the base (lr 1e-6, 10k steps).
- **E3:** SFT on k clean demos of each U task (2k steps, lr 1e-4).

## Files

On the cluster, under /iris/u/lucyshi/fpl_transfer/ktask/:

- `results/e1.png`, `results/e2.png`: figures.
- `results/e1_tables.md`, `results/e1_slopes.md`, `results/e2_tables.md`, `results/e3_table.md`: tables.
- `results/tasks_expert.mp4`: the 32 tasks with the scripted expert.
- `runs/`: every run (specs, logs, checkpoints, evaluations).

Code: `diffusion_policy/ktask/` on the branch transfer-experiments.
