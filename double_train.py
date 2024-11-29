from pong_env import PongDoublePlayerEnv
from agents.dqn_agent import DQNAgent
from agents.random_agent import RandomAgent
from agents.ac_agent import ACAgent
from agents.double_dqn import DoubleDQNAgent
import numpy as np

def create_env():
    return PongDoublePlayerEnv()

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
        agent = ACAgent(
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
    
SCREEN_WIDTH = 156

def transform_obs(observation):
    # for second agent in double obs
    ballx, bally, left, right, speedx, speedy = observation

    ballx = SCREEN_WIDTH - ballx
    left, right = right, left
    speedx = -speedx

    return np.array([ballx, bally, left, right, speedx, speedy])

def run(conf={'num_episodes': 150}, save_path = None, model_paths = None, types = ['rand', 'rand']):
    
    env = create_env()
    train_agent = create_agent(conf, env, agent=types[0], model_path=model_paths[0] if model_paths else None)
    opposing_agent = create_agent(conf, env, agent=types[1], model_path=model_paths[1] if model_paths else None)
    return_list = []
    best_return = float('-inf')
    
    print("Starting Training...")
    for episode in range(conf['num_episodes']):
        cum_return = 0.0
        observation = env.reset()
        train_agent.reset()
        opposing_agent.reset()
        
        done = False
        
        while not done:
            # Select action
            train_action = train_agent.act(observation)
            opponent_action = opposing_agent.act(transform_obs(observation), greedy=True)
            
            # Take action in environment
            next_observation, rewards, done, _ = env.step((train_action, opponent_action))
            train_reward = rewards[0]

            # Learn from experience
            train_agent.learn(train_reward, next_observation, done)

            # update rewards
            observation = next_observation
            cum_return += train_reward
        
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

        if avg_return > 0:
            break
    if save_path:
        train_agent.save_model(save_path)
    
    env.close()
    return return_list

if __name__ == "__main__":
    conf = {
        "num_episodes": 1000
        }
    run(conf = conf, save_path = "models/m3.pth", model_paths=["models/ddqn_single.pth", "models/dqn_single.pth"], types = ['ddqn', 'dqn'])
