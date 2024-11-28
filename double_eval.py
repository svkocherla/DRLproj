from pong_env import PongDoublePlayerEnv
from agents.dqn_agent import DQNAgent
from agents.random_agent import RandomAgent
import numpy as np

def create_env():
    return PongDoublePlayerEnv()

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
    agent1 = create_agent(conf, env, agent = "rand")
    agent2 = create_agent(conf, env, agent = "rand")
    return_list = []
    
    print("Evaluating...")
    for episode in range(conf['num_episodes']):
        cum_return1, cum_return2 = 0.0, 0.0
        observation = env.reset()
        agent1.reset()
        agent2.reset()
        
        # Store first observation
        done = False
        
        while not done:
            # Select action
            action1 = agent1.act(observation)
            action2 = agent2.act(observation)
            
            # Take action in environment
            next_observation, reward, done, _ = env.step((action1, action2))
            
            observation = next_observation
            reward1, reward2 = reward
            cum_return1 += reward1
            cum_return2 += reward2
        
        # Store episode return
        return_list.append((cum_return1, cum_return2))
    
    env.close()
    return return_list

if __name__ == "__main__":
    returns = run()
    returns = np.array(returns)
    print(np.mean(returns, axis = 0)) # shoudl give 2 outputs
    # print(returns.T)