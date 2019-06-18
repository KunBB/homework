import model
import tensorflow as tf
import gym
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 256
LEARNING_RATE = 0.001
HIDDEN_SIZE1 = 128
HIDDEN_SIZE2 = 256
HIDDEN_SIZE3 = 64
SKIP_STEP = 100
NUM_TRAIN = 20000 // BATCH_SIZE
NUM_EPOCH = 10
DROPOUT = 0


def simulate(model, envname, num_rollouts=20, max_timesteps=None, render=False):
    env = gym.make(envname)
    max_steps = max_timesteps or env.spec.timestep_limit

    returns = []
    observations = []
    actions = []
    for i in range(num_rollouts):
        # print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = model.predict(obs.reshape(1, -1))
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if render:
                env.render()
            # if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))
    return np.mean(returns)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, default='/home/yunkunxu/Documents/GitHub/homework/hw1/expert_data/HalfCheetah-v2.pkl')
    parser.add_argument('--envname', type=str, default='HalfCheetah-v2')
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int, default=1000)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    envname = args.envname
    file_name = args.file_name
    bc = model.BehaviorCloning(file_name,
                               envname,
                               HIDDEN_SIZE1,
                               HIDDEN_SIZE2,
                               HIDDEN_SIZE3,
                               BATCH_SIZE,
                               DROPOUT,
                               LEARNING_RATE,
                               SKIP_STEP,
                               True)
    bc.build_graph()
    mean_rewards = []
    for i in range(NUM_EPOCH):
        num_train = NUM_TRAIN
        print("the num_train_step is", num_train)
        bc.train(num_train)
        r = simulate(bc, envname, args.num_rollouts, args.max_timesteps, args.render)
        mean_rewards.append(r)

    plt.figure(envname + '-s2')
    plt.plot(range(1, NUM_EPOCH + 1, 1), mean_rewards)
    plt.xlabel('num_iteration')
    plt.ylabel('average reward')
    plt.savefig(envname + '-s2')
    # plt.show()


if __name__ == '__main__':
    main()