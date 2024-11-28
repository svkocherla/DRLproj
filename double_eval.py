from pong_env import PongDoublePlayerEnv
from agents.dqn_agent import DQNAgent
from agents.random_agent import RandomAgent
import numpy as np

def create_env():
    return PongDoublePlayerEnv()

def create_agent(conf=None, env=None, agent="dqn", model_path=None):
    if agent == "dqn":
        agent = DQNAgent(
            action_space=env.action_space,
            observation_space=env.observation_space,
            **conf
        )
        if model_path:
            agent.load_model(model_path)
        return agent
    else:
        return RandomAgent(
            action_space=env.action_space,
            observation_space=env.observation_space,
        )

SCREEN_WIDTH, SCREEN_HEIGHT = 160, 210

def transform_obs(observation):
    # for second agent in double obs
    ballx, bally, left, right, speedx, speedy = observation

    ballx = SCREEN_WIDTH - ballx
    left, right = right, left
    speedx = -speedx

    return np.array([ballx, bally, left, right, speedx, speedy])

def run(conf=None, model_paths=None):
    if conf is None:
        conf = {'num_episodes': 1000}
    
    env = create_env()
    agent1 = create_agent(conf, env, agent="dqn", model_path=model_paths[0] if model_paths else None)
    agent2 = create_agent(conf, env, agent="dqn", model_path=model_paths[1] if model_paths else None)
    return_list = []
    
    print("Evaluating...")
    for episode in range(conf['num_episodes']):
        cum_return1, cum_return2 = 0.0, 0.0
        observation = env.reset()
        agent1.reset()
        agent2.reset()
        
        done = False
        
        while not done:
            # Select action
            action1 = agent1.act(observation)
            action2 = agent2.act(transform_obs(observation))
            
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
    returns = run() # run(model_paths=["path/to/model1.pth", "path/to/model2.pth"])
    returns = np.array(returns)
    print(np.mean(returns, axis = 0))