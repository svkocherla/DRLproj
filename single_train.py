from pong_env import PongSinglePlayerEnv
from agents.dqn_agent import DQNAgent
from agents.random_agent import RandomAgent
import numpy as np

def create_env():
    return PongSinglePlayerEnv()

def create_agent(conf=None, env=None, agent = "dqn"):
    if agent == "dqn":
        return DQNAgent(
            action_space=env.action_space,
            observation_space=env.observation_space
        )
    else:
        return RandomAgent(
            action_space=env.action_space, 
            observation_space=env.observation_space,
        )

def run(conf=None, save_path=None):
    if conf is None:
        conf = {'num_episodes': 100}
    
    env = create_env()
    agent = create_agent(conf, env, agent = "dqn")
    return_list = []
    best_return = float('-inf')
    
    print("Starting Training...")
    for episode in range(conf['num_episodes']):
        cum_return = 0.0
        observation = env.reset()
        agent.reset()
        
        done = False
        
        while not done:
            # Select action
            action = agent.act(observation)
            
            # Take action in environment
            next_observation, reward, done, _ = env.step(action)
            
            # Learn from experience
            agent.learn(reward, next_observation, done)
            
            observation = next_observation
            cum_return += reward
        
        # Store episode return
        return_list.append(cum_return)
        
        # Calculate running average
        if len(return_list) >= 50:
            avg_return = np.mean(return_list[-50:])
        else:
            avg_return = np.mean(return_list)
        
        # Update best return
        if cum_return > best_return:
            best_return = cum_return
        
        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{conf['num_episodes']}")
            print(f"Return: {cum_return:.2f}")
            print(f"50-episode average: {avg_return:.2f}")
            print(f"Best return: {best_return:.2f}")
            print("--------------------")
    
    if save_path:
        agent.save_model(save_path)

    env.close()   
    return return_list

if __name__ == "__main__":
    run(save_path="models/dqn_single.pth")
    # run()