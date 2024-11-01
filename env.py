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
        
        
        self.physicsClient = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load environment assets
        self.planeId = p.loadURDF("plane.urdf")
        self.tableId = p.loadURDF("table/table.urdf", basePosition=[0, 0, 0])
        self.robotId = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0.6], useFixedBase=True)
        self.bottleId = p.loadURDF("small_bottle.urdf", basePosition=[0.8, 0, 0.65])

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float64)  # 7-DoF + gripper control
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float64)
        
        self.max_steps = 50
        self.current_step = 0
        # Reset environment
        self.reset()

    def reset(self, seed=None, options=None):
        # Reset robot and object positions
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self.planeId = p.loadURDF("plane.urdf")
        self.tableId = p.loadURDF("table/table.urdf", basePosition=[0, 0, 0])
        self.robotId = p.loadURDF("franka_panda/panda.urdf", basePosition=[-0.2, 0, 0.6], useFixedBase=True)
        
        # Randomly place the bottle within a specified range
        bottle_x = np.random.uniform(0.2, 0.6)  # Adjust range based on environment
        bottle_y = np.random.uniform(-0.2, 0.2)  # Adjust range based on environment
        self.bottleId = p.loadURDF("small_bottle.urdf", basePosition=[bottle_x, bottle_y, 0.65])

        self.current_step = 0
        return self._get_obs(), {}

    def step(self, action):
        # Apply action
        joint_positions = action[:7]  # first 7 values for the robot joints
        gripper_action = action[7]    # last value for gripper control

        self.contacts = p.getContactPoints(self.robotId, self.bottleId)
        self.gripper_joint_indices = [9, 10]

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
        self.current_step += 1  # Increment step count
        truncated = self.current_step >= self.max_steps
        info = {}
        

        return obs, reward, done, truncated, info

    def _get_obs(self):
        # Return observation
        robot_pos, robot_ori = p.getBasePositionAndOrientation(self.robotId)
        bottle_pos, bottle_ori = p.getBasePositionAndOrientation(self.bottleId)
        obs = np.array(list(robot_pos) + list(robot_ori) + list(bottle_pos) + list(bottle_ori))
        return obs

    def _compute_reward(self):
        # Get positions for end-effector and bottle
        end_effector_pos = p.getLinkState(self.robotId, 11)[0]
        bottle_pos, _ = p.getBasePositionAndOrientation(self.bottleId)
        distance = np.linalg.norm(np.array(end_effector_pos) - np.array(bottle_pos))

        # Proximity Reward: Reward for moving close to the bottle
        proximity_reward = -distance
        if distance < 0.05:  # Successful grasping threshold
            proximity_reward += 1.0  # Additional reward for very close proximity

        # Grasping Reward: Check if gripper is closed and in contact with the bottle
        gripper_closed = np.all([p.getJointState(self.robotId, idx)[0] < 0.02 for idx in self.gripper_joint_indices])
        grasp_reward = 0
        if gripper_closed and len(self.contacts) > 0:
            grasp_reward = 20.0  # Significant reward for successful grasping

        # Lifting Reward: Reward for lifting the bottle above a certain height
        self.lift_threshold = 0.7  # Define lift threshold
        lifting_reward = 0
        if bottle_pos[2] > self.lift_threshold:
            lifting_reward = 15.0  # Base reward for lifting
            if bottle_pos[2] > self.lift_threshold + 0.1:  # Reward for lifting higher
                lifting_reward += 10.0

        # Encourage movement in joint 6 (wrist rotation) after a certain number of steps  
        joint6_reward = 0      
        if self.current_step > 30:
            joint_6_position = p.getJointState(self.robotId, 6)[0]
            joint_6_movement_reward = -np.abs(joint_6_position) * 0.1  # Penalize lack of movement in joint 6
            joint6_reward += joint_6_movement_reward
        # Total reward
        total_reward = proximity_reward + grasp_reward + lifting_reward + joint6_reward

        return total_reward

    def _is_done(self):
        bottle_position, _ = p.getBasePositionAndOrientation(self.bottleId)
        # 1. what do we want the robot to do for this object?
        if bottle_position[2] > 1:
            print("Episode done: Successfully lifted the object!")
            return True

        # # 2. Check if maximum steps have been reached since we for sure dont want it taking longer than this
        # if self.current_step >= self.max_steps:
        #     print("Episode done: Reached maximum steps without success.")
        #     return True

        # 3. Optional: Add more conditions if needed (e.g., detecting collisions or failures)

        return False

    def render(self, mode="rgb_array"):
        if mode == "rgb_array":
            
            view_matrix = p.computeViewMatrix(cameraEyePosition=[1, 1, 1], cameraTargetPosition=[0, 0, 0], cameraUpVector=[0, 0, 1])
            projection_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=1, nearVal=0.1, farVal=100.0)
            
            
            width, height, rgbImg, depthImg, segImg = p.getCameraImage(width=320, height=320, viewMatrix=view_matrix, projectionMatrix=projection_matrix)
            
            
            return np.array(rgbImg)
        else:
            super().render(mode=mode)

    def close(self):
        p.disconnect()
