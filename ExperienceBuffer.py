import numpy as np

class ExperienceBuffer:
    def __init__(self, buffer_size=2048):
        """ Data structure used to hold game experiences """
        # Buffer will contain [state,action,reward,next_state,done]
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        """ Adds list of experiences to the buffer """
        # Extend the stored experiences
        self.buffer.extend([experience])
        # Keep the last buffer_size number of experiences
        self.buffer = self.buffer[-self.buffer_size:]

    def add_from_lists(self, action_list, state_list, discounted_reward_list):
        for a_id in range(len(action_list)):
            experience_record = np.array([state_list[a_id],
                                          action_list[a_id],
                                          discounted_reward_list[a_id]])
            self.add(experience_record)

    def sample(self, size):
        """ Returns a sample of experiences from the buffer """
        sample_idxs = np.random.randint(len(self.buffer), size=size)
        sample_output = [self.buffer[idx] for idx in sample_idxs]
        sample_output = np.reshape(sample_output, (size, -1))
        return sample_output
