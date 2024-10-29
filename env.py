import pybullet as p
import pybullet_data
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import matplotlib.pyplot as plt

class HealthcareManipulationEnv(gym.Env):
    def __init__(self):
        super(HealthcareManipulationEnv, self).__init__()
        
        # Initialize PyBullet
        self.physicsClient = p.connect(p.GUI)  # Use GUI for visualization
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load environment assets
        self.planeId = p.loadURDF("plane.urdf")
        self.tableId = p.loadURDF("table/table.urdf", basePosition=[0, 0, 0])
        self.robotId = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, -0.5, 0.6], useFixedBase=True)
        
        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float64)  # 7-DoF + gripper control
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float64)

        # Reset environment
        self.reset()

    def reset(self, seed=None, options=None):
        # Reset robot and object positions
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self.planeId = p.loadURDF("plane.urdf")
        self.tableId = p.loadURDF("table/table.urdf", basePosition=[0, 0, 0])
        self.robotId = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0.6], useFixedBase=True)
        self.bottleId = p.loadURDF("small_bottle.urdf", basePosition=[0.3, 0, 0.65])

        # Initial observation
        return self._get_obs(), {}

    def step(self, action):
        # Apply action
        joint_positions = action[:7]  # first 7 values for the robot joints
        gripper_action = action[7]    # last value for gripper control
        
        for i in range(7):
            p.setJointMotorControl2(self.robotId, i, p.POSITION_CONTROL, targetPosition=joint_positions[i])

        # Update gripper control
        if gripper_action > 0:
            p.setJointMotorControl2(self.robotId, 9, p.POSITION_CONTROL, targetPosition=0.04)  # Open
            p.setJointMotorControl2(self.robotId, 10, p.POSITION_CONTROL, targetPosition=0.04)
        else:
            p.setJointMotorControl2(self.robotId, 9, p.POSITION_CONTROL, targetPosition=0.0)  # Close
            p.setJointMotorControl2(self.robotId, 10, p.POSITION_CONTROL, targetPosition=0.0)

        # Step simulation
        p.stepSimulation()
        obs = self._get_obs()

        # Compute reward
        reward = self._compute_reward()
        done = self._is_done()
        info = {}

        return obs, reward, done, done, info

    def _get_obs(self):
        # Return observation
        robot_pos, robot_ori = p.getBasePositionAndOrientation(self.robotId)
        bottle_pos, bottle_ori = p.getBasePositionAndOrientation(self.bottleId)
        obs = np.array(list(robot_pos) + list(robot_ori) + list(bottle_pos) + list(bottle_ori))
        return obs

    def _compute_reward(self):
        # Calculate reward based on distance and task success
        end_effector_pos = p.getLinkState(self.robotId, 11)[0]
        bottle_pos, _ = p.getBasePositionAndOrientation(self.bottleId)
        distance = np.linalg.norm(np.array(end_effector_pos) - np.array(bottle_pos))
        
        # Reward for closer distance and successful grasping
        reward = -distance
        if distance < 0.05:  # Successful grasping threshold
            reward += 1.0
        return reward

    def _is_done(self):
        # Define termination condition
        return False

    def render(self, mode="rgb_array"):
        if mode == "rgb_array":
            # Define camera settings
            view_matrix = p.computeViewMatrix(cameraEyePosition=[1, 1, 1], cameraTargetPosition=[0, 0, 0], cameraUpVector=[0, 0, 1])
            projection_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1, nearVal=0.1, farVal=100.0)
            
            # Capture image
            width, height, rgbImg, depthImg, segImg = p.getCameraImage(width=320, height=320, viewMatrix=view_matrix, projectionMatrix=projection_matrix)
            
            # Return RGB image array
            return np.array(rgbImg)
        else:
            super().render(mode=mode)

    def close(self):
        p.disconnect()