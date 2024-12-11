from ASV_ADMM import ASV_ADMM, ConsensusADMM, BregmanConsensusADMM
import numpy as np

import matplotlib.pyplot as plt

from helpers import sq_norm

import cv2
import imageio
import os


class ResourceSimulation:

    def __init__(self, ADMM_class, X_MIN, X_MAX, M, N, T, NOISE, falloff_rate, seed, video_folder, obstacle_centers=None, obstacle_radii=None):
        
        np.random.seed(seed)

        self.video_folder = video_folder
        frames_folder = os.path.join(self.video_folder, "frames")
        if not os.path.exists(frames_folder):
            os.makedirs(frames_folder)

        self.X_MIN = X_MIN
        self.X_MAX = X_MAX
        self.M = M
        self.N = N
        self.T = T
        self.NOISE = NOISE
        self.falloff_rate = falloff_rate

        self.theta_star = self.gen_theta_star()

        self.obstacle_centers = obstacle_centers or []
        self.obstacle_radii = obstacle_radii or []

        self.x_hist = np.array([self.x_init(self.N)]).reshape((1, self.N, 2))
        self.y_hist = np.array([self.sample_environment(self.x_hist[-1], self.theta_star)]).reshape((1, self.N, 1))
        self.loss_hist = []

        def make_J_i(i):
            return lambda theta_i_flat: self.J_i_func(
                theta_i_flat, self.x_hist[:, i, :], self.y_hist[:, i, :]
            )

        Jis = np.array([make_J_i(i) for i in range(self.N)])

        self.admm = ADMM_class(Jis, self.N, self.p_func, self.x_update_func, self.theta_init(self.N), self.x_init(self.N))

        self.k = 0


    def p_func(self, x_i, x_j):
        ##### communication falloff
        # return 1 / (1 + self.falloff_rate * sq_norm(x_i - x_j))
    
        ##### altitude communication falloff based on mountain

        # # grab the midpoint between the agents
        # midpoint = (x_i + x_j) / 2

        # # if the midpoint is higher on the mountain, have a worse chance of communicating
        # top_am = np.max(self.theta_star[:, 0])
        # midpoint_altitude = self.resource_func(self.theta_star)(midpoint)

        # # print(f"{np.clip((midpoint_altitude / top_am), 0, 1)}")
        # return np.clip((midpoint_altitude / top_am), 0, 1)


        ##### dead zones
        # for center, radius in zip(self.obstacle_centers, self.obstacle_radii):
        #     if sq_norm(center - x_i) < radius ** 2 or sq_norm(center - x_j) < radius ** 2:
        #         return 0
        # return 1

        ##### obstacles
        # if the line between the two agents intersects with an obstacle, return 0
        for center, radius in zip(self.obstacle_centers, self.obstacle_radii):
            # vector from x_i to x_j
            v = x_j - x_i
            # vector from x_i to the center of the obstacle
            w = center - x_i
            c1 = np.dot(w, v)
            c2 = np.dot(v, v)
            b = c1 / c2
            if b < 0:
                closest_point = x_i
            elif b > 1:
                closest_point = x_j
            else:
                closest_point = x_i + b * v

            if sq_norm(closest_point - center) < radius ** 2:
                return 0
        return 1
    
    def in_obstacle(self, x):
        for center, radius in zip(self.obstacle_centers, self.obstacle_radii):
            if sq_norm(center - x) < radius ** 2:
                return True
        return False


    def x_update_func(self, x):
        # x is a 2xN array
        # Each agent takes a random step
        y = self.sample_environment(x, self.theta_star).reshape((1, self.N, 1))

        x = x.reshape((1, self.N, 2))

        self.x_hist = np.append(self.x_hist, x, axis=0)
        self.y_hist = np.append(self.y_hist, y, axis=0)
        
        stepsize = 1
        new_points = np.clip((1 - stepsize) * x + stepsize * ((self.X_MAX - self.X_MIN) * np.random.rand(1, self.N, 2) + self.X_MIN), self.X_MIN, self.X_MAX).reshape((self.N, 2))
        
        # if new points are in an obstacle, don't move there
        for i in range(self.N):
            if self.in_obstacle(new_points[i]):
                new_points[i] = x[0, i]
        
        return new_points

    #Let X_i be all possible states for agent i, given by the square [self.X_MIN, self.X_MAX]^2. 
    # We want each agent to roam the region and take samples

    def x_init(self, num_agents):
        # initialize at random points in the square
        return (self.X_MAX - self.X_MIN) * np.random.rand(num_agents, 2) + self.X_MIN

    def theta_init(self, num_agents):
        return (np.random.rand(num_agents, self.M, 4)).reshape((num_agents, -1))

    # resource availability function - sum of self.M gaussians with amplitude am, center cm, and spread sm
    # theta is Mx3 array (ams, cms, sms)
    def resource_func(self, theta_flat):
        theta = theta_flat.reshape((self.M, 4))
        ams = theta[:, 0]
        cms = theta[:, 1:3]
        sms = theta[:, 3]

        def eval_point(x):
            return np.sum([am * np.exp(-sq_norm(x - cm) / (2 * sm ** 2)) for am, cm, sm in zip(ams, cms, sms)])

        eps = 1e-6 # for stability
        def func(X):
            X = X.reshape(-1, 2)
            # Compute differences and norms in a vectorized way
            diff = X[:, None, :] - cms[None, :, :]  # Shape: (self.N, self.M, 2)
            sq_diffs = sq_norm(diff, axis=2)    # Shape: (self.N, self.M)

            # Compute Gaussian contributions
            gaussians = ams * np.exp(-sq_diffs / (2 * sms ** 2 + eps))  # Shape: (self.N, self.M)

            # Sum contributions from all components
            res = np.sum(gaussians, axis=1)  # Shape: (self.N,)

            return res.reshape(-1, 1)
        
        return func

    # theta is (ams, cmxs, cmys, sms)_i for i=1^self.N. We use MSE as the regression cost function
    def J_i_func(self, theta_i_flat, x_i_hist, y_i_hist):
        f_i = self.resource_func(theta_i_flat)
        residual = (f_i(x_i_hist) - y_i_hist)

        return sq_norm(residual) / x_i_hist.shape[0]

    def sample_environment(self, X, theta_star):
        f_star = self.resource_func(theta_star)
        return f_star(X) + np.random.normal(0, self.NOISE, size=(self.N,1))


    def gen_theta_star(self):
        ams = np.random.rand(self.M) * (self.X_MAX - self.X_MIN) / 4
        cms = (self.X_MAX - self.X_MIN) * np.random.rand(self.M, 2) + self.X_MIN
        sms = np.random.rand(self.M) * (self.X_MAX - self.X_MIN) / 3 + 0.5

        print(f"ams: {ams}\ncms: {cms}\nsms: {sms}")
        return np.array([ams, cms[:, 0], cms[:, 1], sms]).T

    def consensus_loss(self, thetas):
        return np.sum([sq_norm(thetas[i] - np.mean(thetas, axis=0))**2 for i in range(self.N)]) / self.N

    def global_loss(self, thetas, x_hist, theta_star):
        f_star = self.resource_func(theta_star)
        cost = 0
        for i in range(self.N):
            f_i = self.resource_func(thetas[i])
            cost += sq_norm(f_i(x_hist[:, i, :]) - f_star(x_hist[:, i, :]))
        return cost / (self.N * x_hist.shape[0])

    def save_heatmap(self):

        thetas = self.admm.theta

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        
        x = np.linspace(self.X_MIN, self.X_MAX, 101)
        y = np.linspace(self.X_MIN, self.X_MAX, 101)
        X, Y = np.meshgrid(x, y)

        # combine X and Y into a single 2xn array
        arr = np.vstack([X.ravel(), Y.ravel()]).T

        Z = np.array([self.resource_func(self.theta_star)(point) for point in arr]).reshape(X.shape)
        axs[0].contourf(X, Y, Z, 20, cmap='cividis')
        axs[0].set_title("Ground Truth")


        # average plots of all the other thetas
        Z = np.zeros_like(X)
        for theta in thetas:
            Z += 1 / self.N * np.array([self.resource_func(theta)(point) for point in arr]).reshape(X.shape)
        
        axs[1].contourf(X, Y, Z, 20, cmap='cividis')

        axs[1].set_title("Average Agent Estimate")

        axs[0].set_xlim(self.X_MIN, self.X_MAX)
        axs[0].set_ylim(self.X_MIN, self.X_MAX)
        axs[1].set_xlim(self.X_MIN, self.X_MAX)
        axs[1].set_ylim(self.X_MIN, self.X_MAX)

        plt.suptitle(f"Iteration {self.k}")
        plt.savefig(f"{self.video_folder}/frames/temp{self.k}.png")
        plt.close()
        
    def plot_loss_curves(self, loss_hist):
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        for i in range(self.N):
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


    def theta_bounds(self):
        out = []
        for i in range(self.M):
            out.append((0, None))
            out.append((self.X_MIN, self.X_MAX))
            out.append((self.X_MIN, self.X_MAX))
            out.append((0, None))
        return out

    def step(self):
        losses = {i : self.J_i_func(self.admm.theta[i], self.x_hist[:, i, :], self.y_hist[:, i, :]) for i in range(self.N)}
        losses["global"] = self.global_loss(self.admm.theta, self.x_hist, self.theta_star)
        losses["consensus"] = self.consensus_loss(self.admm.theta)
        self.loss_hist.append(losses)

        self.admm.update(self.theta_bounds())

        self.k += 1


def save_comparison_heatmap(theta_star, theta_asv, theta_cadmm, resource_func, x_min, x_max, k, image_folder, obstacle_centers=None, obstacle_radii=None):
    fig, axs = plt.subplots(1, 3, figsize=(16, 9))

    og_axs = axs
    axs = axs.flat

    theta_asv_mean = np.mean(theta_asv, axis=0)
    theta_cadmm_mean = np.mean(theta_cadmm, axis=0)

    x = np.linspace(x_min, x_max, 101)
    y = np.linspace(x_min, x_max, 101)
    X, Y = np.meshgrid(x, y)

    # combine X and Y into a single 2xn array
    arr = np.vstack([X.ravel(), Y.ravel()]).T

    Z_gt = np.array([resource_func(theta_star)(point) for point in arr]).reshape(X.shape)
    vmin = np.min(Z_gt)
    vmax = np.max(Z_gt)

    im = axs[0].contourf(X, Y, Z_gt, 20, cmap='cividis', vmin=vmin, vmax=vmax)
    # plot obstacles on ground truth map

    obstacle_centers = obstacle_centers or []
    obstacle_radii = obstacle_radii or []
    for center, radius in zip(obstacle_centers, obstacle_radii):
        circle = plt.Circle(center, radius, color='blue', fill=True)
        axs[0].add_artist(circle)

    axs[0].set_title("Ground Truth")

    Z_ASV = np.array([resource_func(theta_asv_mean)(point) for point in arr]).reshape(X.shape)

    axs[1].contourf(X, Y, Z_ASV, 20, cmap='cividis', vmin=vmin, vmax=vmax)
    axs[1].set_title("ASV-ADMM Average Agent Estimate")

    Z_CADMM = np.array([resource_func(theta_cadmm_mean)(point) for point in arr]).reshape(X.shape)
    axs[2].contourf(X, Y, Z_CADMM, 20, cmap='cividis', vmin=vmin, vmax=vmax)
    axs[2].set_title("Consensus-ADMM Average Agent Estimate")

    for ax in axs:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(x_min, x_max)
        # ensure each plot is square
        ax.set_aspect('equal')



    mappable = plt.cm.ScalarMappable(cmap='cividis')
    mappable.set_array(Z_gt)
    mappable.set_clim(vmin, vmax)
    
    fig.colorbar(mappable, ax=og_axs.ravel().tolist(), orientation='horizontal')

    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    plt.suptitle(f"Iteration {k}")
    plt.savefig(f"{image_folder}/temp{k}.png")

    plt.close()

def save_loss_curves(num_agents, loss_hist_asv, loss_hist_cadmm, image_folder):

    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    # leave more gap between the plots
    fig.subplots_adjust(wspace=0.25)

    # have the first two plots share y
    axs[0].sharey(axs[1])


    alpha=1/num_agents

    for ax in axs:
        ax.set_yscale('log')
        ax.set_xlabel("Iteration")

    for (loss_dict_asv, loss_dict_cadmm) in zip(loss_hist_asv, loss_hist_cadmm):
        for i in range(num_agents):
            # plot all the asv agents in one color and the cadmm agents in another color, plot it with light opacity so as to not muddy the plot

            axs[0].plot([loss_dict_asv[i] for loss_dict_asv in loss_hist_asv], label=f"Agent {i}", color='blue', alpha=alpha, linewidth=0.25)
            axs[0].plot([loss_dict_cadmm[i] for loss_dict_cadmm in loss_hist_cadmm], label=f"Agent {i}", color='red', alpha=alpha, linewidth=0.25)

        axs[1].plot([loss_dict_asv["global"] for loss_dict_asv in loss_hist_asv], color='blue', alpha=1)
        axs[1].plot([loss_dict_cadmm["global"] for loss_dict_cadmm in loss_hist_cadmm], color='red', alpha=1)

        axs[2].plot([loss_dict_asv["consensus"] for loss_dict_asv in loss_hist_asv], color='blue', alpha=1)
        axs[2].plot([loss_dict_cadmm["consensus"] for loss_dict_cadmm in loss_hist_cadmm], color='red', alpha=1)

    # custom legend of blue being asv, red being cadmm. put it above the entire figure. make it fully opaque now
    fig.legend(["ASV-ADMM", "Consensus-ADMM"], loc='upper center', ncol=2)

    axs[0].set_title("Local Losses")
    axs[1].set_title("Global Loss")
    axs[2].set_title("Consensus Loss")


    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    plt.savefig(f"{image_folder}/loss-curve.png")
    plt.show()
    plt.close()

def stitch_video(video_filename, frames_folder, T, frame_size, fps=30):
    cv2out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps=fps, frameSize=frame_size)
    for k in range(T):
        img = cv2.imread(f"{frames_folder}/temp{k}.png")
        img = cv2.resize(img, frame_size)
        cv2out.write(img)
    cv2out.release()
    cv2.destroyAllWindows()

def stitch_gif(gif_filename, frames_folder, T, fps=30):
    images_bgr = [cv2.imread(f"{frames_folder}/temp{k}.png") for k in range(T)]
    images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images_bgr]
    imageio.mimsave(gif_filename, images_rgb, fps=fps)

def gen_obstacles(X_MIN, X_MAX, num_obstacles):
    obstacle_centers = [(X_MAX - X_MIN) * np.random.rand(2) + X_MIN for _ in range(num_obstacles)]
    obstacle_radii = [0.1 + 0.2 * np.random.rand() for _ in range(num_obstacles)]
    return obstacle_centers, obstacle_radii

def main():

    # X_MIN=-1, X_MAX=1, M=3, N=10, T=1000, NOISE=0.1, falloff_rate=1, seed=79 is a good run

    # X_MIN=-1, X_MAX=1, M=1, N=3, T=200, NOISE=0.1, falloff_rate=3, NOISE=0.1, seed=79 differentiates between ours and baseline

    X_MIN = -1
    X_MAX = 1
    M = 1
    N = 5
    T = 50
    NOISE = 0.1
    falloff_rate = 3

    seed = 82

    obstacle_centers, obstacle_radii = gen_obstacles(X_MIN, X_MAX, 5)

    RUN_NAME = f"realobs_BregmanComparisonX_MIN={X_MIN}_X_MAX={X_MAX}_M={M}_N={N}_T={T}_NOISE={NOISE}_falloff_rate={falloff_rate}_seed={seed}_obs={len(obstacle_centers)}"

    asv = ResourceSimulation(ASV_ADMM, X_MIN, X_MAX, M, N, T, NOISE, falloff_rate, seed, video_folder="resource-sim-asv", obstacle_centers=obstacle_centers, obstacle_radii=obstacle_radii)
    cadmm = ResourceSimulation(BregmanConsensusADMM, X_MIN, X_MAX, M, N, T, NOISE, falloff_rate, seed, video_folder="resource-sim-cadmm", obstacle_centers=obstacle_centers, obstacle_radii=obstacle_radii)
    for k in range(T):
        asv.step()
        cadmm.step()

        save_comparison_heatmap(asv.theta_star, asv.admm.theta, cadmm.admm.theta, asv.resource_func, X_MIN, X_MAX, k, f"{RUN_NAME}/frames", obstacle_centers=obstacle_centers, obstacle_radii=obstacle_radii)

        if (k % 20 == 0):

            print("\n\nIteration", k)
            print(f"ASV-ADMM Local losses at iteration {k}: {[asv.J_i_func(asv.admm.theta[i], asv.x_hist[:, i, :], asv.y_hist[:, i, :]) for i in range(N)]}")
            print(f"ASV-ADMM Global loss at iteration {k}: {asv.global_loss(asv.admm.theta, asv.x_hist, asv.theta_star)}")
            print(f"ASV-ADMM Consensus loss at iteration {k}: {asv.consensus_loss(asv.admm.theta)}")
            print("\n")
            print(f"Consensus-ADMM Local losses at iteration {k}: {[cadmm.J_i_func(cadmm.admm.theta[i], cadmm.x_hist[:, i, :], cadmm.y_hist[:, i, :]) for i in range(N)]}")
            print(f"Consensus-ADMM Global loss at iteration {k}: {cadmm.global_loss(cadmm.admm.theta, cadmm.x_hist, cadmm.theta_star)}")
            print(f"Consensus-ADMM Consensus loss at iteration {k}: {cadmm.consensus_loss(cadmm.admm.theta)}")
            print("\n")

        
    save_loss_curves(N, asv.loss_hist, cadmm.loss_hist, f"{RUN_NAME}/losses")
    stitch_video(f"{RUN_NAME}/video.mp4", f"{RUN_NAME}/frames", 200, (1280, 720))
    # baseline = ResourceSimulation(ConsensusADMM, video_folder="resource-sim-frames-baseline")


if __name__ == "__main__":
    main()
