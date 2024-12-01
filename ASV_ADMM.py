import scipy.optimize
import numpy as np

from helpers import sq_norm

class ASV_ADMM:

    def __init__(self, Jis, num_agents, p_func, x_update_func, theta_init, x_init):
        # N
        self.num_agents = num_agents
        
        # 1xN array of cost functions, one for each agent
        self.J = Jis
        assert (self.J.shape[0] == self.num_agents)

        # decision variables, R^(N x n) (one row for each agent)
        self.theta = theta_init
        assert (len(self.theta.shape) == 2)
        assert (self.theta.shape[0] == self.num_agents)

        # auxiliary states, R^(N x m) (one row for each agent)
        self.x = x_init
        assert (len(self.x.shape) == 2)
        assert (self.x.shape[0] == self.num_agents)
        
        # maps states x_i, x_j to communication probabilities 
        self._p_func = p_func
        # updates x
        self._x_update_func = x_update_func
        
        self._update_p()
        
        # dual variable aggregate, Nxn
        self.e = np.zeros_like(self.theta)

        # nonstatic communication graph. use adjacency matrix formulation
        self._neighbors = np.random.sample((self.num_agents, self.num_agents)) < self.p
        self._update_neighbors()

        # quadratic penalty parameter
        self.rho = 0.001

        # bregman divergence parameter
        self.beta0 = 1000

        self.timestep = 1

    def neighbors(self, i):
        return np.argwhere(self._neighbors[i])
    
    def alpha(self, i, j):
        denom = 0
        for l in self.neighbors(i):
            denom += np.exp(self.p[i, l])
        return np.exp(self.p[i, j]) / denom

    def update(self, theta_bounds=None):

        next_theta = np.copy(self.theta)

        # primal update - can be done in parallel
        for i in range(self.num_agents):

            def theta_i_func(theta_i):
                obj = self.J[i](theta_i) + self.e[i].T @ theta_i
                for j in self.neighbors(i):
                    obj += self.rho * self.alpha(i, j) * sq_norm(theta_i - (self.theta[i] + self.theta[j]) / 2) \
                           + np.sqrt(self.timestep) / self.beta0 * sq_norm(theta_i - self.theta[i])
                return obj
            
            optim_result = scipy.optimize.minimize(theta_i_func, 
                                                   self.theta[i], 
                                                   bounds=theta_bounds, 
                                                   method='L-BFGS-B')
            if (optim_result.success):
                next_theta[i] = optim_result.x
            else:
                print(f"Optimization failed: {optim_result.message}")


        next_e = np.copy(self.e)
        
        # dual update - can be done in parallel
        for i in range(self.num_agents):
            for j in self.neighbors(i):
                # next_e[i] += self.rho * self.alpha(i, j) * (next_theta[i] - next_theta[j]).reshape(-1)
                next_e[i] = self.e[i]

        # state update
        self._update_x()


        # graph update
        self._update_p()
        self._update_neighbors()

        self.theta = next_theta
        self.e = next_e

        self.timestep += 1

    def _update_x(self):
        self.x = self._x_update_func(self.x)

    def _update_p(self):
        self.p = np.zeros((self.num_agents, self.num_agents))

        for i in range(self.num_agents):
            for j in range(self.num_agents):
                self.p[i, j] = self._p_func(self.x[i], self.x[j])


    def _update_neighbors(self):
        self._neighbors = np.random.sample(self._neighbors.shape) < self.p
