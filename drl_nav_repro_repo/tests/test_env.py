import numpy as np

from sim.envs.tb3_nav_env import TB3NavEnv


def test_env_step_runs():
    env = TB3NavEnv()
    obs, info = env.reset(seed=0)
    assert obs.shape[0] == 126
    for _ in range(10):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        assert np.isfinite(r)
