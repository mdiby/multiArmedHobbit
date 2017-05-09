import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_c_reward(rewards, c_list=None, cust=None):
    'Ingests a reward dictionary and outputs the cummulative regret'
    sns.set_style("whitegrid")
    plt.figure(figsize=(13, 7))
    c_rewards = pd.DataFrame(rewards)
    if c_list: #if vector containing the customer type is passed in
      # filter only for customers of a specific type
        lst = np.array(np.where(np.array(c_list) == cust)).flatten()
        c_rewards = c_rewards.iloc[lst, :]

    c_rewards = c_rewards.apply(np.cumsum)
    for col in c_rewards:
        plt.plot(c_rewards[col], label=col)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('iterations', fontsize=14)
    plt.ylabel('reward', fontsize=14)
    plt.title('Cummulative Reward')
    plt.show()


def plot_c_regret(rewards, c_list=None, cust=None):
    'Ingests a reward dictionary and outputs the cummulative regret'
    sns.set_style("whitegrid")
    plt.figure(figsize=(13,7))
    c_rewards = pd.DataFrame(rewards)

    if c_list:  # if vector containing the customer type is passed in
      # filter only for customers of a specific type
        lst = np.array(np.where(np.array(c_list) == cust)).flatten()
        c_rewards = c_rewards.iloc[lst,:]

    c_rewards = c_rewards.apply(np.cumsum)

    for col in c_rewards:
        if col is not 'wtp':
            plt.plot(c_rewards['wtp'] - c_rewards[col], label=col)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel('iterations', fontsize=14)
    plt.ylabel('reward', fontsize=14)
    plt.title('Cummulative Regret')
    plt.show()


def plot_arm_loss(loss, smooth=5):
    'Ingests a reward dictionary and outputs the cummulative regret'
    sns.set_style("white")
    # plt.figure(figsize=(13,7))
    f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True, sharey=True, figsize=(14, 10))
    loss = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in loss.items()]))
    loss = pd.ewma(loss, span=smooth)

    ax1.plot(loss[1], label='Arm 1', color='r')
    ax1.legend(loc="upper right")
    ax2.plot(loss[2], label='Arm 2', color='g')
    ax2.legend(loc="upper right")
    ax3.plot(loss[3], label='Arm 3', color='m')
    ax3.legend(loc="upper right")
    ax4.plot(loss[4], label='Arm 4', color='y')
    ax4.legend(loc="upper right")
    ax5.plot(loss[5], label='Arm 5', color='b')
    ax5.legend(loc="upper right")
    ax1.set_title('Loss by Arm', fontsize=14)
    ax5.set_xlabel('iterations', fontsize=12)
    ax3.set_ylabel('loss', fontsize=12)

    f.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

    plt.show()
