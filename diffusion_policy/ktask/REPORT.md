# Does RL on K tasks help held-out tasks, and does that change with K? (ManiSkill3, 2026-10-14)

All runs complete:

- E1: 54 PPO runs.
- E1x: 12 precise evaluations of the K = 1 training tasks.
- E2: 30 offline arms.
- E3: 12 learnability runs.

Changes are success rates vs the base, paired on start states. Brackets are 95% CIs over runs and start
states.

## Short answer

1. **Yes, PPO on K training tasks raises success on held-out tasks at every K.**

   | K | runs | training tasks | H (held out) | U (never seen) |
   |---|---|---|---|---|
   | 1 | 12 | +0.14 [+0.07, +0.22] | +0.08 [+0.05, +0.11] | +0.09 [+0.04, +0.14] |
   | 2 | 12 | +0.14 [+0.09, +0.19] | +0.08 [+0.06, +0.10] | +0.09 [+0.05, +0.13] |
   | 4 | 12 | +0.12 [+0.08, +0.17] | +0.10 [+0.08, +0.12] | +0.10 [+0.06, +0.15] |
   | 8 | 12 | +0.11 [+0.07, +0.15] | +0.10 [+0.08, +0.12] | +0.10 [+0.07, +0.15] |
   | 16 | 6 | +0.11 [+0.06, +0.15] | +0.09 [+0.06, +0.11] | +0.10 [+0.05, +0.15] |

   H was in pretraining but never trained with RL. U are object–target pairs that were never in
   pretraining. All runs use 32,768 episodes in total.

2. **What changes with K is how much of the gain transfers, and how reliably.** Below is the held-out change
   per unit of training-task change: each run's slope over its 4k–32k checkpoints, then the mean over runs.

   | K | runs | H per unit | U per unit | runs above 0.5 |
   |---|---|---|---|---|
   | 1 | 12 | 0.67 [0.40, 0.94] | 0.61 [0.33, 0.90] | 7/12 |
   | 2 | 12 | 0.61 [0.47, 0.80] | 0.63 [0.45, 0.79] | 7/12 |
   | 4 | 12 | 0.87 [0.70, 1.05] | 0.83 [0.65, 1.01] | 11/12 |
   | 8 | 12 | 1.00 [0.87, 1.14] | 0.85 [0.71, 1.00] | 12/12 |
   | 16 | 6 | 0.90 [0.73, 1.06] | 0.95 [0.88, 1.03] | 6/6 |

   - From K = 4 on, about 0.9 of the training gain appears on held-out tasks, in almost every run.
   - With 1–2 tasks it is about 0.6–0.7 on average, and erratic. One single-task run (puck on the
     platform, +0.47 on its task) transferred as broadly as K = 16; others hardly moved held-out tasks.
   - The episode budget is fixed, so the training-task gain shrinks a little as K grows. The absolute
     held-out gain therefore rises from +0.08 (K ≤ 2) to +0.10 (K = 4–8) and stays there.

3. **The update rule matters, mainly when K is small** (E2: the same tasks, each arm from the base). Cells
   are training change, then held-out change (H + rest of R).

   | | PPO, 32k episodes | SFT on clean expert demos, 10k steps | filtered BC on PPO's own successes, 10k steps |
   |---|---|---|---|
   | K = 1 (6 runs) | +0.16, **+0.07** | +0.15, **−0.09** | +0.01, **−0.18** |
   | K = 4 (6 runs) | +0.13, +0.11 | +0.15, +0.10 | −0.03, −0.07 |
   | K = 16 (3 runs) | +0.11, +0.09 | +0.13, +0.13 | −0.04, −0.05 |

   - On a single task, supervised fine-tuning makes the policy forget the other tasks. This happens even
     with the same episodes PPO collected, as on robosuite. PPO's update instead improves them.
   - With 4–16 tasks, SFT on expert demos transfers as well as PPO.
   - Filtered BC on PPO's episodes is worse than the base at every K, including on the training tasks.
     I don't have an explanation for that yet.

4. **RL does not make new tasks easier to learn** (E3).
   - After SFT on 5 or 20 clean demos of each never-seen task, every start reaches 0.69–0.71 on those tasks.
     This holds from the base and from PPO on 1, 4 or 16 tasks; pairwise differences are ≤ 0.003.
   - The PPO starts' zero-shot head start (0.58–0.61 vs 0.50) disappears.

5. **Speed does not change, so there is nothing to transfer.** Steps to success moved by at most 1.6%
   in every group; sparse-reward PPO here makes the policy succeed more, not faster.

## What transfers

- **Shared skills: object and target gains recombine.**
  - With K = 1, held-out tasks that share the trained task's object gained +0.16 [+0.08, +0.26]; those
    sharing the target, or neither, gained +0.07.
  - In the K = 16 pilot (seed 1), every puck task gained about +0.4, including the held-out one
    (0.16 → 0.58) and the never-seen one (0.23 → 0.58).
  - In the same pilot, the never-seen cube-on-goal went 0.16 → 0.66. Its grasp and placement had each been
    trained only in other pairs.
  - The tee and the tall box barely moved, trained or not.
- **Both phases.** On held-out tasks, the lift rate rose by +0.05–0.06 and success after a lift by
  +0.07–0.09.
- **A broadly shared weakness can be fixed from one task.** On the weakest base (seed 1, 0.41 on R), PPO on
  puck-on-platform alone moved the whole family the way K = 16 did (puck +0.49, goal placements +0.33).
  - More training tasks make transfer reliable because some task in the set is likely to expose what is
    broadly wrong. This fits the spread at K ≤ 2, but I haven't tested it directly.

## Caveats

- **One task family.** All demos come from the same scripted operator with the same noise model, so the
  base's failure modes are partly shared by construction, which favors broad transfer. On robosuite, where
  the round nut needed its own grasp, PPO on A hurt B. Transfer should be tested where failure modes are
  task-specific.
- **Fixed total budget.** Per-task budgets differ across K: 32k episodes for K = 1, 2k per task for
  K = 16.
- **K = 16 replicates.** K = 16 always trains the same 16 tasks, so its 6 runs vary only in base and PPO
  seeds.
- **K = 1 measurement noise.** With 128 episodes, one task's estimate has a standard error of about 0.05.
  So K = 1 training changes in the slope analysis use 512-episode evaluations (E1x).
- **PPO setting.** It came from pilots: critic warmup, lr 3e-6, 1,024 episodes per iteration. Larger
  per-iteration steps first collapsed the policy, then recovered.

## Setup

- **Tasks:** 32 = 8 objects × 4 targets on the ManiSkill3 Panda tabletop (GPU physics), split by a Latin
  design:
  - U: 8 tasks, never in pretraining;
  - H: 8 tasks, held out from RL;
  - R: 16 tasks, eligible for RL.
- **Base policy:** the robosuite flow-matching transformer, trained with BC on mixed-quality scripted demos
  of R ∪ H (73% successful). Three seeds, at 0.57, 0.41 and 0.62 success on R.
- **E1:** PPO (DPPO-style, as on robosuite) on nested, balanced chains of K tasks from R.
  - 4 chains × 3 base seeds for K < 16; 2 PPO seeds × 3 base seeds for K = 16.
  - Every checkpoint is evaluated on all 32 tasks, 128 episodes each, on the base's start states.
- **E2:** two offline arms from the base on the same tasks (lr 1e-6): filtered BC on PPO's logged
  successful episodes, and SFT on clean expert demos.
- **E3:** SFT on k ∈ {5, 20} clean demos per U task (2,000 steps, lr 1e-4).
- **Design notes:** DESIGN.md.

## Robosuite follow-up (all seeds now in)

- **a3, PPO on B after 5 demos, round nut:** 0.18 → 0.30 from the A-only base, 0.16 → 0.30 after PPO on A,
  and 0.19 → 0.34 after SFT on A.
- **xtime, second pair with a time cost:** PPO on A gives A +0.18, B −0.18. A gets 9.8% faster; B's
  speed is unchanged (+1.0%).
- Conclusions unchanged.

## Files

On the cluster, under /iris/u/lucyshi/fpl_transfer/ktask/:

- `results/e1.png`, `results/e2.png`: figures.
- `results/e1_tables.md`, `results/e1_slopes.md`, `results/e2_tables.md`, `results/e3_table.md`: tables.
- `results/tasks_expert.mp4`: the 32 tasks with the scripted expert.
- `runs/`: specs, logs, checkpoints and evaluations of every run.

Code: `diffusion_policy/ktask/` on the branch transfer-experiments.
