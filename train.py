from pong_env import PongSinglePlayerEnv
from agents.dqn_agent import DQNAgent
import numpy as np

def create_env():
    return PongSinglePlayerEnv()

def create_agent(conf=None, env=None):
    if conf is None:
        conf = {}
    
    default_conf = {
        'learning_rate': 1e-4,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_final': 0.01,
        'epsilon_decay': 10000,
        'memory_size': 10000,
        'batch_size': 64,
        'target_update': 10
    }
    
    # Update default configuration with provided configuration
    for key in conf:
        if key in default_conf:
            default_conf[key] = conf[key]
    
    return DQNAgent(
        action_space=env.action_space,
        observation_space=env.observation_space,
        **default_conf
    )

def run(conf=None):
    if conf is None:
        conf = {'num_episodes': 1000}
    
    env = create_env()
    agent = create_agent(conf, env)
    return_list = []
    best_return = float('-inf')
    
    print("Starting Training...")
    for episode in range(conf['num_episodes']):
        cum_return = 0.0
        observation = env.reset()
        agent.reset()  # Reset agent's episode-specific variables
        
        # Store first observation
        agent.last_observation = observation
        done = False
        
        while not done:
            # Select action
            action = agent.act(observation)
            agent.last_action = action
            
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
            print(f"Epsilon: {agent.get_epsilon():.3f}")
            print("--------------------")
    
    env.close()
    return return_list

if __name__ == "__main__":
    config = {
        'num_episodes': 5000,
        'learning_rate': 1e-3,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_final': 0.01,
        'epsilon_decay': 20000,
        'memory_size': 10000,
        'batch_size': 64,
        'target_update': 10
    }
    run(config)