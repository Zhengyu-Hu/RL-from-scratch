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

def quality(R, Q, n):
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

if __name__ == '__main__':
    # set random seed
    RANDOM = 1
    if RANDOM == 0:
        np.random.seed(42) 

    bandit = Bandit()
    agent = Agent()
    steps = 1000
    total_reward = 0
    total_rewards = []
    rates = []
    actions = []

    # Start to play
    for i in range(steps):
        action = agent.get_action()
        reward = bandit.play(action)
        agent.update(reward, action)
        total_reward += reward

        actions.append(action)
        total_rewards.append(total_reward)
        rates.append(total_reward / (i+1)) 



    # Show results
    import matplotlib.pyplot as plt

    plt.figure('Results',figsize=[10,5])

    plt.subplot(121)
    plt.ylabel('Total Rewards')
    plt.xlabel('Steps')
    plt.plot(total_rewards)

    plt.subplot(122)
    plt.ylabel('Rates')
    plt.xlabel('Steps')
    plt.plot(rates)

    values, counts = np.unique(actions, return_counts=True)
    idx = np.argmax(counts)
    most_common = values[idx]
    freq = counts[idx]

    print(f'Most common action is {most_common}, which is chosen by agent {freq} times!')
    print(bandit.rates)

    plt.show()
