"""Where does an env step's time go? (state obs vs none, evaluate, controller, physics)"""
import time, torch, gymnasium as gym
import ktask.env
from ktask.env import N_TASKS

def bench(obs_mode, n=1024, steps=50, **kw):
    ids = [i % N_TASKS for i in range(n)]
    env = gym.make('KTaskPlace-v0', num_envs=n, task_ids=ids, obs_mode=obs_mode, control_mode='pd_ee_pose',
                   sim_backend='gpu', render_backend='none', **kw)
    u = env.unwrapped
    env.reset(seed=list(range(n)))
    a = torch.zeros(n, 7, device=u.device)
    a[:, :3] = u.agent.tcp.pose.p - u.agent.robot.pose.p
    a[:, 3] = 3.14159
    a[:, 6] = 1
    for _ in range(5): env.step(a)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(steps): env.step(a)
    torch.cuda.synchronize(); dt = (time.time() - t) / steps
    # pieces
    torch.cuda.synchronize(); t = time.time()
    for _ in range(steps): u.evaluate()
    torch.cuda.synchronize(); de = (time.time() - t) / steps
    torch.cuda.synchronize(); t = time.time()
    for _ in range(steps): u.agent.set_action(a); u.agent.before_simulation_step()
    torch.cuda.synchronize(); dc = (time.time() - t) / steps
    torch.cuda.synchronize(); t = time.time()
    for _ in range(steps): u.scene.step()
    torch.cuda.synchronize(); dp = (time.time() - t) / steps
    torch.cuda.synchronize(); t = time.time()
    for _ in range(steps): u.policy_obs()
    torch.cuda.synchronize(); do = (time.time() - t) / steps
    print(f'{obs_mode:6s} n={n}: step {dt*1e3:.1f} ms | evaluate {de*1e3:.1f} | controller {dc*1e3:.1f} | one physics substep {dp*1e3:.1f} (x{u._sim_steps_per_control}) | policy_obs {do*1e3:.1f}', flush=True)
    env.close()

for mode in ('state', 'none'):
    try:
        bench(mode)
    except Exception as e:
        print(mode, 'failed:', repr(e)[:200])
bench('none', n=256)
bench('none', n=4096)
