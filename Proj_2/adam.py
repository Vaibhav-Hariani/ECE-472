from tensorflow import math
class Adam():
    def __init__(
        self,
        size,
        beta_1=0.9,
        beta_2=0.999,
        step_size = 0.001,
        epsilon = 1e-8,
    ):
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.step_size = step_size
        self.epsilon = epsilon
        self.t = 0
        self.size = size
        self.m_ts = [0] * size
        self.v_ts = [0] * size

    def train(self,grads,vars):
        self.t += 1
        for i in range(self.size):
            self.m_ts[i] = self.beta_1 * self.m_ts[i] + (1-self.beta_1) * grads[i]
            self.v_ts[i] = self.beta_2 * self.v_ts[i] + (1-self.beta_2) * (grads[i] * grads[i])
            self.step_size = self.step_size * math.sqrt(1-self.beta_2**self.t)/(1-self.beta_1**self.t)
            offset = (self.step_size * self.m_ts[i] / math.sqrt(self.v_ts[i] + self.epsilon))
            vars[i].assign_sub(offset)
