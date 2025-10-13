import gym

# Create the environment
env = gym.make("CartPole-v1", render_mode="human")

# Reset the environment to start a new episode
observation, info = env.reset(seed=42)

for _ in range(500):  # run for 500 steps
    action = env.action_space.sample()   # take a random action
    observation, reward, done, truncated, info = env.step(action)
    if done or truncated:
        observation, info = env.reset()
        
env.close()
