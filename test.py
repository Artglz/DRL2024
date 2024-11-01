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
    # print the timestep
    print(f"Time Step: {env.current_step}")
    if terminated or truncated:
        obs, _ = env.reset()
    
    env.render()

# Close the environment
env.close()

# from stable_baselines3 import PPO
# from env import HealthcareManipulationEnv

# # Load the trained model
# model = PPO.load("healthcare_manipulation_ppo_model")

# # Create a new environment instance for evaluation
# eval_env = HealthcareManipulationEnv()

# # Run a few episodes to evaluate the model
# num_episodes = 10
# for episode in range(num_episodes):
#     obs = eval_env.reset()
#     done = False
#     total_reward = 0
#     while not done:
#         action, _ = model.predict(obs)
#         obs, reward, done, truncated, _ = eval_env.step(action)
#         total_reward += reward
#     print(f"Episode {episode + 1}: Total Reward = {total_reward}")
