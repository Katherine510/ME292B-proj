from ASV_ADMM import ASV_ADMM, BaselineADMM
import numpy as np

import matplotlib.pyplot as plt

from helpers import sq_norm

import cv2


def p_func(x_i, x_j):
    # communication falloff
    falloff_rate = 1
    return 1 / (1 + falloff_rate * sq_norm(x_i - x_j))
    # return 1


#Let X_i be all possible states for agent i, given by the square [X_MIN, X_MAX]^2. 
# We want each agent to roam the region and take samples

def x_init(num_agents):
    # initialize at random points in the square
    return (X_MAX - X_MIN) * np.random.rand(num_agents, 2) + X_MIN

# resource availability function - sum of M gaussians with amplitude am, center cm, and spread sm
# theta is Mx3 array (ams, cms, sms)
def resource_func(theta_flat):
    theta = theta_flat.reshape((M, 4))
    ams = theta[:, 0]
    cms = theta[:, 1:3]
    sms = theta[:, 3]

    def eval_point(x):
        return np.sum([am * np.exp(-sq_norm(x - cm) / (2 * sm ** 2)) for am, cm, sm in zip(ams, cms, sms)])

    eps = 1e-6 # for stability
    def func(X):
        X = X.reshape(-1, 2)
        # Compute differences and norms in a vectorized way
        diff = X[:, None, :] - cms[None, :, :]  # Shape: (N, M, 2)
        sq_diffs = sq_norm(diff, axis=2)    # Shape: (N, M)

        # Compute Gaussian contributions
        gaussians = ams * np.exp(-sq_diffs / (2 * sms ** 2 + eps))  # Shape: (N, M)

        # Sum contributions from all components
        res = np.sum(gaussians, axis=1)  # Shape: (N,)

        return res.reshape(-1, 1)
    
    return func

# theta is (ams, cmxs, cmys, sms)_i for i=1^N. We use MSE as the regression cost function
def J_i_func(theta_i_flat, x_i_hist, y_i_hist):
    f_i = resource_func(theta_i_flat)
    residual = (f_i(x_i_hist) - y_i_hist)

    return sq_norm(residual) / x_i_hist.shape[0]

def sample_environment(X, theta_star):
    f_star = resource_func(theta_star)
    return f_star(X) + np.random.normal(0, NOISE, size=(N,1))


def gen_theta_star():
    np.random.seed(79)
    ams = np.random.rand(M) * (X_MAX - X_MIN) / 4
    cms = (X_MAX - X_MIN) * np.random.rand(M, 2) + X_MIN
    sms = np.random.rand(M) * (X_MAX - X_MIN) / 3 + 0.5

    print(f"ams: {ams}\ncms: {cms}\nsms: {sms}")
    return np.array([ams, cms[:, 0], cms[:, 1], sms]).T


def theta_init(num_agents):
    return (np.random.rand(num_agents, M, 4)).reshape((num_agents, -1))

def consensus_loss(thetas):
    return np.sum([sq_norm(thetas[i] - np.mean(thetas, axis=0))**2 for i in range(N)]) / N

def global_loss(thetas, x_hist, theta_star):
    f_star = resource_func(theta_star)
    cost = 0
    for i in range(N):
        f_i = resource_func(thetas[i])
        cost += sq_norm(f_i(x_hist[:, i, :]) - f_star(x_hist[:, i, :]))
    return cost / (N * x_hist.shape[0])

def heatmap(thetas, theta_star, k):

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
    # for i in range(N):
    #     axs[0].scatter(admm_x[i, 0], admm_x[i, 1], c='red')
    #     axs[1].scatter(admm_x[i, 0], admm_x[i, 1], c='red')

    axs[1].set_title("Average Agent Estimate")

    axs[0].set_xlim(X_MIN, X_MAX)
    axs[0].set_ylim(X_MIN, X_MAX)
    axs[1].set_xlim(X_MIN, X_MAX)
    axs[1].set_ylim(X_MIN, X_MAX)

    plt.suptitle(f"Iteration {k}")
    plt.savefig(f"frames/temp{k}.png")
    plt.close()
    
def plot_loss_curves(loss_hist):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    for i in range(N):
        axs[0].plot([loss_dict[i] for loss_dict in loss_hist], label=f"Agent {i}")
    axs[0].set_title("Local Losses")
    axs[0].legend()

    axs[1].plot([loss_dict["global"] for loss_dict in loss_hist])
    axs[1].set_title("Global Loss")

    axs[2].plot([loss_dict["consensus"] for loss_dict in loss_hist])
    axs[2].set_title("Consensus Loss")

    # log scale on y axis
    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    axs[2].set_yscale('log')
    
    plt.show()


def theta_bounds():
    out = []
    for i in range(M):
        out.append((0, None))
        out.append((X_MIN, X_MAX))
        out.append((X_MIN, X_MAX))
        out.append((0, None))
    return out

def stitch_video():
    image_folder = 'frames'
    video_name = 'vid2.mp4'

    cv2out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (1000, 500))
    for k in range(T):
        img = cv2.imread(f"frames/temp{k}.png")
        cv2out.write(img)
    cv2out.release()
    cv2.destroyAllWindows()


X_MIN = -1
X_MAX = 1
M = 3
N = 10
T = 500
NOISE = 0.03

def main():
    theta_star = gen_theta_star()

    x_hist = np.array([x_init(N)]).reshape((1, N, 2))
    y_hist = np.array([sample_environment(x_hist[-1], theta_star)]).reshape((1, N, 1))
    loss_hist = []
    
    def x_update_func(x):
        # x is a 2xN array
        # Each agent takes a random step
        y = sample_environment(x, theta_star).reshape((1, N, 1))

        x = x.reshape((1, N, 2))

        nonlocal x_hist
        nonlocal y_hist
        x_hist = np.append(x_hist, x, axis=0)
        y_hist = np.append(y_hist, y, axis=0)
        
        stepsize = 1
        return np.clip((1 - stepsize) * x + stepsize * ((X_MAX - X_MIN) * np.random.rand(1, N, 2) + X_MIN), X_MIN, X_MAX).reshape((N, 2))


    Jis = np.array([
        lambda theta_i_flat : J_i_func(theta_i_flat, x_hist[:, i, :], y_hist[:, i, :])
        for i in range(N)
    ])
    admm = ASV_ADMM(Jis, N, p_func, x_update_func, theta_init(N), x_init(N))
    for k in range(T):
        losses = {i : J_i_func(admm.theta[i], x_hist[:, i, :], y_hist[:, i, :]) for i in range(N)}
        losses["global"] = global_loss(admm.theta, x_hist, theta_star)
        losses["consensus"] = consensus_loss(admm.theta)
        loss_hist.append(losses)

        if (k % 20 == 0):
            print("Iteration", k)
            print(f"Local losses at iteration {k}: {[J_i_func(admm.theta[i], x_hist[:, i, :], y_hist[:, i, :]) for i in range(N)]}")
            print(f"Global loss at iteration {k}: {global_loss(admm.theta, x_hist, theta_star)}")
            print(f"Consensus loss at iteration {k}: {consensus_loss(admm.theta)}")
        if (k % 1 == 0):
            # print(admm.theta.reshape((N, M, 4)))
            heatmap(admm.theta, theta_star, k)
            # plot_loss_curves(loss_hist)
        admm.update(theta_bounds())

    stitch_video()

if __name__ == "__main__":
    main()
