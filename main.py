import os
import gymnasium as gym
from env import CustomEnv
from game_env import GameEnv

from sb3_plus import MultiOutputPPO

game_env = GameEnv(
                    max_steps= 256,
                    n_colors= 3,
                    n_agents= 9,
                    n_pieces= 9,
                    n_letters= 2,
                    grid_size = 12)
env = CustomEnv(env=game_env)

model_path = "saves/coolmodel.save"

model = MultiOutputPPO(policy='MIMOPolicy', env=env, verbose=1)
if not os.path.exists(model_path):
    model.save(model_path)

for _ in range(1000):
    game_env.init_instances(model.load(model_path))

    model.learn(total_timesteps=2048 * 5, progress_bar=True)
    model.save(model_path)

"""
# The ideia is train the agent together with n of its clones and update the clones with the new knowladge every k steps

obs = env.reset()

for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
"""