# from tensorflow import math, identity

from tensorflow import math


class Adam:
    def __init__(
        self,
        size,
        beta_1=0.9,
        beta_2=0.999,
        step_size=0.001,
        epsilon=1e-7,
        w=0.004,
    ):
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.step_size = step_size
        self.epsilon = epsilon
        self.t = 0
        self.size = size
        self.m_ts = [0] * size
        self.v_ts = [0] * size
        self.w = w

    def train(self, grads, vars, adamW=False, decay_scale=0.1):
        self.t += 1
        for i in range(self.size):
            self.m_ts[i] = self.beta_1 * self.m_ts[i] + (1 - self.beta_1) * grads[i]
            self.v_ts[i] = self.beta_2 * self.v_ts[i] + (1 - self.beta_2) * (
                grads[i] * grads[i]
            )
            m_t_hat = self.m_ts[i] / (1 - self.beta_1**self.t)
            v_t_hat = self.v_ts[i] / (1 - self.beta_2**self.t)
            offset = self.step_size * m_t_hat / math.sqrt(v_t_hat + self.epsilon)
            if adamW:
                offset += self.w * vars[i]
            offset = decay_scale * offset
            vars[i].assign_sub(offset)

    def AdaMax(self, grads, vars):
        self.t += 1
        for i in range(self.size):
            grad = grads[i] + vars[i] * self.w
            self.m_ts[i] = self.beta_1 * self.m_ts[i] + (1 - self.beta_1) * grad
            # For AdaMax, v_t becomes u_t according to the paper
            self.u_ts[i] = (math.max(self.beta_2 * self.v_ts[i], math.abs(grad)),)
            offset = (
                (self.step_size / (1 - self.beta_1**self.t))
                * self.m_ts[i]
                / self.u_ts[i]
            )
            vars[i].assign_sub(offset)
