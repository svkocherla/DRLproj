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
            observation_space=env.observation_space,
            **conf
        )
    else:
        return RandomAgent(
            action_space=env.action_space, 
            observation_space=env.observation_space,
        )
    
def run(conf=None):
    if conf is None:
        conf = {'num_episodes': 1000}
    
    env = create_env()
    agent = create_agent(conf, env, agent = "dqn")
    return_list = []
    
    print("Evaluating...")
    for episode in range(conf['num_episodes']):
        cum_return = 0.0
        observation = env.reset()
        agent.reset()
        
        # Store first observation
        done = False
        
        while not done:
            # Select action
            action = agent.act(observation)
            
            # Take action in environment
            next_observation, reward, done, _ = env.step(action)
            
            # rewards and obs
            observation = next_observation
            cum_return += reward
        
        # Store episode return
        return_list.append(cum_return)
    
    env.close()
    return return_list

if __name__ == "__main__":
    returns = run()
    returns = np.array(returns)
    print(np.mean(returns))