from env import HealthcareManipulationEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt

# Training the PPO agent
env = HealthcareManipulationEnv()
check_env(env)

# Track rewards for visualization
rewards = []

# Train the PPO agent
model = PPO("MlpPolicy", env, verbose=1)

# Modify learn to collect rewards
def callback(locals, globals):
    global rewards
    if 'rewards' in locals:
        rewards.append(locals['rewards'])
    return True

model.learn(total_timesteps=10000, callback=callback)  # Increase timesteps for a full experiment

# Save the model
model.save("healthcare_manipulation_ppo")

# Evaluate the model
obs, _ = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
    env.render()
env.close()

# Plotting the results
plt.plot(range(len(rewards)), rewards)
plt.title('Cumulative Rewards Over Episodes')
plt.xlabel('Episodes')
plt.ylabel('Cumulative Reward')
plt.show()