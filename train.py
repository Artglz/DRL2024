from stable_baselines3 import SAC
from env import HealthcareManipulationEnv
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import pybullet as p
import os
import matplotlib.pyplot as plt

class RewardCallback(BaseCallback):
    """
    Custom callback for tracking rewards during training.
    """

    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.rewards = []  # List to store rewards

    def _on_step(self) -> bool:
        # Get current reward from the last step
        if 'rewards' in self.locals:
            self.rewards.append(self.locals['rewards'])
        return True  # Keep training

def main():
    # Initialize environment
    train_env = HealthcareManipulationEnv()
    train_env = DummyVecEnv([lambda: train_env])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False, clip_obs=10.)

    # Hyperparameters
    hyperparameters = {
        "learning_rate": 4e-4,        # SAC's default learning rate
        "buffer_size": int(1e6),      # Large replay buffer size
        "batch_size": 32,              # Batch size for training
        "tau": 0.005,                 # Soft update coefficient for target networks
        "gamma": 0.99,                # Discount factor for rewards
        "total_timesteps": 10000,     # Total number of timesteps to train the agent
        "verbose": 1                  # Verbosity level (0: no output, 1: info)
    }

    policy_kwargs = dict(net_arch=[256, 256])

    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=hyperparameters['learning_rate'],
        buffer_size=hyperparameters['buffer_size'],
        batch_size=hyperparameters['batch_size'],
        tau=hyperparameters['tau'],
        gamma=hyperparameters['gamma'],
        policy_kwargs=policy_kwargs,
        verbose=hyperparameters['verbose'],
    )

    eval_callback = EvalCallback(
        train_env,
        eval_freq=500,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    # Instantiate the reward callback
    reward_callback = RewardCallback()

    # Train the agent
    model.learn(
        total_timesteps=hyperparameters['total_timesteps'],
        callback=[eval_callback, reward_callback],
        log_interval=10
    )

    # Save the trained model and normalization stats
    model.save("sac_panda_model")
    train_env.save("vec_normalize.pkl")

    train_env.close()

    # Plotting the training rewards
    plt.figure(figsize=(12, 6))
    plt.plot(reward_callback.rewards, label='Cumulative Rewards', color='b')
    plt.title('Cumulative Rewards Over Episodes')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
