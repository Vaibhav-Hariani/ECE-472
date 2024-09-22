from tensorflow import math, identity

class Adam:
    # Custom Adam implementation
    def __init__(
        self, size, beta_1=0.9, beta_2=0.999, step_size=0.001, epsilon=1e-8, w=0,in_func=identity
    ):
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.step_size = step_size
        self.epsilon = epsilon
        self.t = 0
        self.w = w
        self.size = size
        self.m_ts = [0] * size
        self.v_ts = [0] * size
        self.in_func=in_func

    def train(self, grads, vars, adamW=False):
        self.t += 1
        for i in range(self.size):
            grad = grads[i]
            if not adamW:
                grad += vars[i] * self.w

            self.m_ts[i] = self.beta_1 * self.m_ts[i] + (1 - self.beta_1) * grad
            self.v_ts[i] = self.beta_2 * self.v_ts[i] + (1 - self.beta_2) * (
                grad * grad
            )
            self.step_size = (
                self.step_size
                * math.sqrt(1 - self.beta_2**self.t)
                / (1 - self.beta_1**self.t)
            )
            epsilon = self.epsilon / math.sqrt(1 - self.beta_2**self.t)
            offset = self.step_size * self.m_ts[i] / math.sqrt(self.v_ts[i] + epsilon)
            if adamW:
                offset += self.w * vars[i]
            offset = self.in_func(offset)
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
