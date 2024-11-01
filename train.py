from env import HealthcareManipulationEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
import time

# Training the PPO agent
env = HealthcareManipulationEnv()
check_env(env)

class RewardTrackingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardTrackingCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.current_rewards = 0

    def _on_step(self) -> bool:
        # Accumulate rewards per step
        self.current_rewards += self.locals['rewards'][0]
        done = self.locals['dones'][0]
        
        # If an episode is done, append the episode's total reward
        if done:
            self.episode_rewards.append(self.current_rewards)
            self.current_rewards = 0
        
        return True  # To continue training

start = time.time()
# Initialize and pass the callback to PPO
reward_callback = RewardTrackingCallback()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000, callback=reward_callback)
end = time.time()
total_time = end - start
# Save the trained model
model.save("healthcare_manipulation_ppo")

# Plot cumulative rewards per episode
plt.plot(range(len(reward_callback.episode_rewards)), reward_callback.episode_rewards)
plt.title(f'Cumulative Rewards Over Episodes (Time Taken: {total_time:.2f} seconds)')
plt.xlabel('Episodes')
plt.ylabel('Cumulative Reward')
plt.show()
