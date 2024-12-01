
from ASV_ADMM import ASV_ADMM
import numpy as np

import matplotlib.pyplot as plt

import time


def p_func(x_i, x_j):
    # communication falloff
    # return X_MAX / (1 + np.linalg.norm(x_i - x_j) ** 2)
    return 1


#Let X_i be all possible states for agent i, given by the square [-X_MIN, X_MAX]^2. 
# We initialize each agent to be equally spaced around the unit circle (i.e. x_i = [cos(2pi * i/N), sin(2pi * i/N)]).
# We want each agent to roam the region and take samples

def x_init(num_agents):
    # return np.array([[np.cos(2 * np.pi * i / num_agents), np.sin(2 * np.pi * i / num_agents)] for i in range(num_agents)])
    return np.array([[0, 0] for i in range(num_agents)])

# resource availability function - sum of M gaussians with amplitude am, center cm, and spread sm
# theta is Mx3 array (ams, cms, sms)
def resource_func(theta_flat):
    theta = theta_flat.reshape((M, 4))
    ams = theta[:, 0]
    cms = theta[:, 1:3]
    sms = theta[:, 3]

    epsilon = 0.001 # for stability
    return lambda x : np.sum([am * np.exp(-np.linalg.norm(x - cm) ** 2 / (2 * sm ** 2 + epsilon)) for am, cm, sm in zip(ams, cms, sms)])

# theta is (ams, cmxs, cmys, sms)_i for i=1^N. We use MSE as the regression cost function
def J_i_func(theta_i_flat, x_i_hist, y_i_hist):

    f_i = resource_func(theta_i_flat)

    cost = 0

    for x, y in zip(x_i_hist, y_i_hist):
        # print(x, f_i(x), y)
        cost += (f_i(x) - y) ** 2

    return cost / x_i_hist.shape[0]

def sample_environment(x, theta_star):
    f_star = resource_func(theta_star)
    res = np.abs(f_star(x) + np.random.normal(0, NOISE, (N, 1)))
    return res


def gen_theta_star():
    np.random.seed(4)
    ams = np.random.rand(M) * 1 + 1
    cms = 2 * X_MAX * np.random.rand(M, 2) - X_MAX
    sms = np.random.rand(M) * 2 + 1

    print(f"ams: {ams}\ncms: {cms}\nsms: {sms}")
    return np.array([ams, cms[:, 0], cms[:, 1], sms]).T


def theta_init(num_agents):
    return (np.random.rand(num_agents, M, 4)).reshape((num_agents, -1))

def global_loss(theta, x_hist, theta_star):
    f_star = resource_func(theta_star)
    cost = 0
    for i in range(N):
        f_i = resource_func(theta[i])
        for x in x_hist[:, i, :]:
            cost += (f_i(x) - f_star(x)) ** 2
    return cost / (N * x_hist.shape[0])

def heatmap(thetas, theta_star, admm_x):

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    x = np.linspace(X_MIN, X_MAX, 101)
    y = np.linspace(X_MIN, X_MAX, 101)
    X, Y = np.meshgrid(x, y)

    # combine X and Y into a single 2xn array
    arr = np.vstack([X.ravel(), Y.ravel()]).T

    Z = np.array([resource_func(theta_star)(point) for point in arr]).reshape(X.shape)
    axs[0].contourf(X, Y, Z, 20, cmap='cividis')
    axs[0].set_title("Actual")


    # average plots of all the other thetas
    Z = np.zeros_like(X)
    for theta in thetas:
        Z += 1 / N * np.array([resource_func(theta)(point) for point in arr]).reshape(X.shape)
    
    axs[1].contourf(X, Y, Z, 20, cmap='cividis')

    # plot points where agents are
    for i in range(N):
        axs[0].scatter(admm_x[i, 0], admm_x[i, 1], c='red')
        axs[1].scatter(admm_x[i, 0], admm_x[i, 1], c='red')

    axs[1].set_title("Average Agent Estimate")

    plt.show()
    
def plot_loss_curves(losses):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    for i in range(N):
        axs[0].plot([loss[i] for loss in losses], label=f"Agent {i}")
    axs[0].set_title("Local Losses")
    axs[0].legend()

    axs[1].plot([loss["global"] for loss in losses])
    axs[1].set_title("Global Loss")

    plt.show()


def theta_bounds():
    out = []
    for i in range(M):
        out.append((0, None))
        out.append((-X_MAX, X_MAX))
        out.append((-X_MAX, X_MAX))
        out.append((0, None))
    return out

X_MIN = -3
X_MAX = 3
M = 1
N = 4
T = 1000
NOISE = 0.0

def main():
    theta_star = gen_theta_star()

    x_hist = np.array([x_init(N)]).reshape((1, N, 2))
    y_hist = np.array([sample_environment(x_hist[-1], theta_star)]).reshape((1, N, 1))
    loss_hist = []

    def x_update_func(x):
        # x is a 2xN array
        # Each agent takes a random step
        x = x.reshape((1, N, 2))
        nonlocal x_hist
        nonlocal y_hist

        x_hist = np.append(x_hist, x, axis=0)
        y = sample_environment(x, theta_star).reshape((1, N, 1))
        y_hist = np.append(y_hist, y, axis=0)
        
        epsilon = 1
        return np.clip((1 - epsilon) * x + epsilon * (2 * X_MAX * np.random.rand(1, N, 2) - X_MAX), X_MIN, X_MAX).reshape((N, 2))


    Jis = np.array([
        lambda theta_i_flat : J_i_func(theta_i_flat, x_hist[:, i, :], y_hist[:, i, :])
        for i in range(N)
    ])
    admm = ASV_ADMM(Jis, N, p_func, x_update_func, theta_init(N), x_init(N))
    for k in range(T):
        losses = {i : J_i_func(admm.theta[i], x_hist[:, i, :], y_hist[:, i, :]) for i in range(N)}
        losses["global"] = global_loss(admm.theta, x_hist, theta_star)
        loss_hist.append(losses)

        if (k % 10 == 0):
            print(f"Local losses at iteration {k}: {[J_i_func(admm.theta[i], x_hist[:, i, :], y_hist[:, i, :]) for i in range(N)]}")
            print(f"Global loss at iteration {k}: {global_loss(admm.theta, x_hist, theta_star)}")
            print(admm.theta.reshape((N, M, 4)))
            heatmap(admm.theta, theta_star, admm.x)
            plot_loss_curves(loss_hist)
        admm.update(theta_bounds())
        # print(admm.x)

if __name__ == "__main__":
    main()
