from pong_env import PongSinglePlayerEnv
from agents.dqn_agent import DQNAgent
from agents.random_agent import RandomAgent
from agents.ac import ActorCriticAgent
from agents.double_dqn import DoubleDQNAgent
import numpy as np

def create_env():
    return PongSinglePlayerEnv()

def create_agent(conf=None, env=None, agent="dqn", model_path=None):
    if agent == "dqn":
        agent = DQNAgent(
            action_space=env.action_space,
            observation_space=env.observation_space,
        )
        if model_path:
            agent.load_model(model_path)
        return agent
    elif agent == "ac":
        agent = ActorCriticAgent(
            action_space=env.action_space,
            observation_space=env.observation_space,
        )
        if model_path:
            agent.load_model(model_path)
        return agent
    elif agent == "ddqn":
        agent = DoubleDQNAgent(
            action_space=env.action_space,
            observation_space=env.observation_space,
        )
        if model_path:
            agent.load_model(model_path)
        return agent
    else:
        return RandomAgent(
            action_space=env.action_space,
            observation_space=env.observation_space,
        )

def run(conf=None, model_path=None):
    if conf is None:
        conf = {'num_episodes': 100}
    
    env = create_env()
    agent = create_agent(conf, env, agent="ddqn", model_path=model_path)
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
            action = agent.act(observation, greedy=True)
            
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
    returns = run(model_path="models/ddqn_single.pth")
    returns = np.array(returns)
    print(np.mean(returns))