from agents.base_agent import BaseAgent

# for testing double player env
class DQNAgent(BaseAgent):
    '''Random agent.'''

    def __init__(self, action_space, observation_space, **kwargs):
        self._action_space = action_space

    def act(self, observation, greedy=False):
        return self._action_space.sample()

    def learn(self, reward, next_observation, done):
        pass

    def save_model(self, path):
        pass

    def load_model(self, path):
        pass