import numpy as np

class DinoAgent:
    """ Dino agent that uses Q-learning to play the game Dino Run.
        Inputs: pixel values of the game screen
        Actions: 0: duck, 1: jump OR 0 - do nothing, 1: duck, 2: jump
        Rewards: 1: game is still running, 0: game is over (episode end)
    """
    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions
        self.q_table = np.zeros((n_states, n_actions))
        
    def act(self, inputs):
        state = self.get_state(inputs)
        return np.argmax(self.q_table[state])

    def get_state(self, inputs):
        """ Convert pixel values to state 
            by converting the inputs to a base 2 number
        """
        state = 0
        for i in range(len(inputs)):
            state += inputs[i] * (2**i)
        return int(state)

if __name__ == '__main__':
    agent = DinoAgent(2**16, 2)
    print(agent.q_table)

    # test get_state
    inputs = [1]*16
    state = agent.get_state(inputs)
    print(state)