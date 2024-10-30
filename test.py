from env import HealthcareManipulationEnv
from stable_baselines3 import PPO

# Load the environment and trained model
env = HealthcareManipulationEnv()
model = PPO.load("healthcare_manipulation_ppo")

# Evaluate the model
obs, _ = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, _ = env.reset()
    
    env.render()

# Close the environment
env.close()
