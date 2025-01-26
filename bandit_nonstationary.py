from banditprob import Bandit, Agent
import numpy as np
from copy import copy

class NonStatBandit(Bandit):
    def __init__(self, arms=10):
        super().__init__(arms)

    def play(self, arm):
        rate = self.rates[arm]

        # 胜率发生变化
        self.rates += 0.1 * np.random.randn(self.arms) # 增加一个标准差为0.1，均值为0的噪声

        if rate > np.random.rand():
            return 1
        else:
            return 0

def EMA(R, Q, alpha):
    return Q + alpha * (R-Q)

class AlphaAgent(Agent):
    def __init__(self, epsilon=0.1, alpha=0.8, action_size=10):
        super().__init__(epsilon, action_size)
        self.alpha = alpha

    def update(self, reward, action):
        self.ns[action] += 1
        self.Qs[action] = EMA(reward, self.Qs[action], self.alpha)

def play(agent, bandit, steps=1000):
    total_reward = 0
    total_rewards = []
    rates = []
    # actions = []

    # Start to play
    for i in range(steps):
        action = agent.get_action()
        reward = bandit.play(action)
        agent.update(reward, action)
        total_reward += reward

        # actions.append(action)
        total_rewards.append(total_reward)
        rates.append(total_reward / (i+1))
    
    return rates

def avg_play(Agent, Env, steps=1000, num_plays=200):
    '''
    这里评价的是平均，所以传入的不应该是一个实例而是一个类。

    每次都需要重新实例化。
    '''
    # all_rewards = np.zeros([num_plays, steps])
    all_rates = np.zeros([num_plays, steps])

    for p in range(num_plays):
        # 注意，需要更新一下老虎机和代理
        bandit = Env()
        agent = Agent()
        all_rates[p] = play(agent, bandit, steps)

    # avg_rewards = np.average(all_rewards, axis=0)
    avg_rates = np.average(all_rates, axis=0)

    return avg_rates

if __name__ == '__main__':
    RANDOM = 0
    if not RANDOM:
        np.random.seed(42)
    rates_eq = avg_play(Agent, NonStatBandit)
    rates_alpha = avg_play(AlphaAgent, NonStatBandit)

    from matplotlib import pyplot as plt
    plt.figure('Eq vs Alpha')

    plt.plot(rates_eq)
    plt.plot(rates_alpha)
    plt.legend(['eq', 'alpha'])
    plt.ylabel('Rates')

    plt.show()
