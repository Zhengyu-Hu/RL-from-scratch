import numpy as np

class Bandit:
    def __init__(self, arms=10):
        # 均匀分布
        self.arms = arms
        self.rates = np.random.rand(10)
    
    def play(self, arm):
        rate = self.rates[arm]
        n = np.random.rand()
        if rate > n:
            return 1
        else:
            return 0

def quality(R, Q=0, n=1):
    return Q + 1/n * (R - Q)


class Agent:
    def __init__(self, epsilon=0.1, action_size=10):
        self.epsilon = epsilon
        self.action_size = action_size
        # action_size表示可以选择的行动的种类数量
        # action指示选择哪一个arm
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)

    def update(self, reward, action):
        self.ns[action] += 1
        self.Qs[action] = quality(reward, self.Qs[action], self.ns[action])
    
    def get_action(self):
        if np.random.rand() > self.epsilon:
            # exploitation
            return np.argmax(self.Qs)
        else:
            # random exploration
            return np.random.randint(0, self.action_size)
        
def run(eps):
    # set random seed
    RANDOM = 1
    if RANDOM == 0:
        np.random.seed(42) 

    bandit = Bandit()
    agent = Agent(epsilon=eps)
    steps = 1000
    total_reward = 0

    rates = []


    # Start to play
    for i in range(steps):
        action = agent.get_action()
        reward = bandit.play(action)
        agent.update(reward, action)
        total_reward += reward
        rates.append(total_reward / (i+1))

    return rates

def eval(eps):
    N = 100
    results = np.zeros([N, 1000])
    for i in range(N):
        rates = run(eps)
        results[i] = rates
    avg_rates = np.average(results, axis=0)
    return avg_rates

eps_list = [0.1, 0.3, 0.01]
all_results = [eval(eps) for eps in eps_list]

import matplotlib.pyplot as plt
plt.figure()

for i in range(len(eps_list)):
    plt.plot(all_results[i])
plt.legend(eps_list)
plt.show()