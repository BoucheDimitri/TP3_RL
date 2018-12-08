import numpy as np
import lqg1d
import matplotlib.pyplot as plt
import utils
import importlib

import funcs
importlib.reload(funcs)
importlib.reload(lqg1d)

# Plot parameters
plt.rcParams.update({"font.size": 30})
plt.rcParams.update({"lines.linewidth": 5})
plt.rcParams.update({"lines.markersize": 10})


#####################################################
# Define the environment and the policy
#####################################################
env = lqg1d.LQG1D(initial_state_type='random')


#####################################################
# Experiments parameters
#####################################################
# We will collect N trajectories per iteration
N = 200
# Each trajectory will have at most T time steps
T = 100
# Number of policy parameters updates
n_itr = 200
# Set the discount factor for the problem
discount = 0.9
# Learning rate for the gradient update
learning_rate = 0.00007
# Number of runs to average on
K = 10


# ######################### Q1: REINFORCE ##############################################################################
# We use an ADAM step

mean_parameters_stacked = []
avg_returns_stacked = []

for k in range(0, K):
    stepper = funcs.AdamStep(alpha=0.1)
    theta = np.random.normal(0, 1)
    policy = funcs.GaussianPolicy(theta)
    mean_parameters = []
    avg_return = []
    grad_estimates = []
    for _ in range(n_itr):
        paths = utils.collect_episodes(env, policy=policy, horizon=T, n_episodes=N)
        avg_return.append(funcs.average_return(paths))
        grad_estimate = 0
        for n in range(0, N):
            # Rtau = funcs_Q1.R_estimate(paths[n]["rewards"], discount)
            # print("Rtau" + str(Rtau))
            grad_traj = funcs.trajectory_gradient(policy,
                                                    paths[n]["states"][:, 0],
                                                    paths[n]["actions"][:, 0],
                                                  paths[n]["rewards"],
                                                  discount)
        # print(grad_traj)
        # grad_estimate += (1 / N) * (Rtau * grad_traj)
            grad_estimate += (1 / N) * grad_traj
        grad_estimates.append(grad_estimate)
        theta = policy.get_theta()
        theta += stepper.update(grad_estimate)
        policy.set_theta(theta)
    mean_parameters_stacked.append(policy.theta_records.copy())
    avg_returns_stacked.append(avg_return)
    print(k)

avg_returns_stacked = np.array(avg_returns_stacked)
avg_traj_returns = avg_returns_stacked.mean(axis=0)
std_traj_returns = avg_returns_stacked.std(axis=0)

mean_parameters_stacked = np.array(mean_parameters_stacked)
avg_mean_parameter = mean_parameters_stacked.mean(axis=0)
std_mean_parameter = mean_parameters_stacked.std(axis=0)

# plot the average return obtained by simulating the policy
# at each iteration of the algorithm (this is a rought estimate
# of the performance
fig, axes = plt.subplots(ncols=2)
axes[0].plot(avg_traj_returns, label="Average")
y1 = avg_traj_returns + std_traj_returns
y2 = avg_traj_returns - std_traj_returns
x = np.arange(0, y1.shape[0])
axes[0].fill_between(x, y1, y2, alpha=0.1)
axes[0].set_ylabel("Average rewards")
axes[0].set_xlabel("Iteration")
axes[0].legend()

# plot the distance mean parameter
# of iteration k
axes[1].plot(avg_mean_parameter + 0.6, label="Average")
y1 = avg_mean_parameter + 0.6 + std_mean_parameter
y2 = avg_mean_parameter + 0.6 - std_mean_parameter
axes[1].fill_between(x, y1, y2, alpha=0.1)
axes[1].set_ylabel("theta - theta*")
axes[1].set_xlabel("Iteration")
axes[1].legend()

plt.suptitle("ADAM paces - Averaged over 10 runs")


# ############# Q2: REINFORCE with Exploration Bonuses #################################################################
# We use an ADAM step
pace_s = 4
pace_a = 4
beta = 30

mean_parameters_stacked = []
avg_returns_stacked = []

for k in range(0, K):
    stepper = funcs.AdamStep(alpha=0.1)
    theta = np.random.normal(0, 1)
    policy = funcs.GaussianPolicy(theta)
    mean_parameters = []
    avg_return = []
    grad_estimates = []
    for _ in range(n_itr):
        paths = utils.collect_episodes(env, policy=policy, horizon=T, n_episodes=N)
        avg_return.append(funcs.average_return(paths))
        grad_estimate = 0
        for n in range(0, N):
            bonuses = funcs.bonus_functions(paths[n]["states"], paths[n]["actions"], beta, pace_s, pace_a)
            grad_traj = funcs.trajectory_gradient(policy,
                                                    paths[n]["states"][:, 0],
                                                    paths[n]["actions"][:, 0],
                                                  paths[n]["rewards"] + bonuses,
                                                  discount)
        # print(grad_traj)
        # grad_estimate += (1 / N) * (Rtau * grad_traj)
            grad_estimate += (1 / N) * grad_traj
        grad_estimates.append(grad_estimate)
        theta = policy.get_theta()
        theta += stepper.update(grad_estimate)
        policy.set_theta(theta)
    mean_parameters_stacked.append(policy.theta_records.copy())
    avg_returns_stacked.append(avg_return)
    print(k)


avg_returns_stacked = np.array(avg_returns_stacked)
avg_traj_returns = avg_returns_stacked.mean(axis=0)
std_traj_returns = avg_returns_stacked.std(axis=0)

mean_parameters_stacked = np.array(mean_parameters_stacked)
avg_mean_parameter = mean_parameters_stacked.mean(axis=0)
std_mean_parameter = mean_parameters_stacked.std(axis=0)

# plot the average return obtained by simulating the policy
# at each iteration of the algorithm (this is a rought estimate
# of the performance
fig, axes = plt.subplots(ncols=2)
axes[0].plot(avg_traj_returns, label="Average")
y1 = avg_traj_returns + std_traj_returns
y2 = avg_traj_returns - std_traj_returns
x = np.arange(0, y1.shape[0])
axes[0].fill_between(x, y1, y2, alpha=0.1)
axes[0].set_ylabel("Average rewards")
axes[0].set_xlabel("Iteration")
axes[0].legend()

# plot the distance mean parameter
# of iteration k
axes[1].plot(avg_mean_parameter + 0.6, label="Average")
y1 = avg_mean_parameter + 0.6 + std_mean_parameter
y2 = avg_mean_parameter + 0.6 - std_mean_parameter
axes[1].fill_between(x, y1, y2, alpha=0.1)
axes[1].set_ylabel("theta - theta*")
axes[1].set_xlabel("Iteration")
axes[1].legend()

plt.suptitle("ADAM paces - Averaged over 10 runs")
