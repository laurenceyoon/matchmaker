import numpy as np


class KalmanPathLattice:
    def __init__(self, size, variance, beat_frames, max_step):
        self.R = variance
        self.F = np.array([[1., 1.], [0., 1.]])
        self.Q = 3 * variance / beat_frames**3 * np.array([[1 / 3, 1 / 2], [1 / 2, 1.]])
        self.steps = np.arange(max_step + 1)
        self.costs = np.full(size, np.inf)
        self.costs[0] = 0.
        self.states = np.column_stack((np.arange(size), np.ones(size)))
        self.covariances = np.tile(np.diag([variance, 1.]), (size, 1, 1))

    def step(self, distances, start, gamma, input_index, horizontal_weight):
        if input_index == 0:
            self.costs[0] = float(np.sum(distances)) / gamma
            return 0, float(distances[0])
        positions = np.arange(start, start + len(distances))
        ancestors = positions[:, None] - self.steps
        valid = ancestors >= start
        ancestors = np.maximum(ancestors, 0)
        states = self.states[ancestors] @ self.F.T
        covariances = self.F @ self.covariances[ancestors] @ self.F.T + self.Q
        innovations = positions[:, None] - states[:, :, 0]
        variances = covariances[:, :, 0, 0] + self.R
        transitions = .5 * (innovations**2 / variances + np.log(2 * np.pi * variances))
        cumulative = np.r_[0., np.cumsum(distances)]
        left = np.maximum(positions[:, None] - self.steps + 1 - start, 0)
        emission = cumulative[np.arange(len(positions))[:, None] + 1] - cumulative[left]
        emission[:, 0] = horizontal_weight * distances
        costs = np.where(valid, self.costs[ancestors] + transitions + emission / gamma, np.inf)
        reachable = np.isfinite(costs).any(axis=1)
        minima = np.min(costs, axis=1)
        minima = np.where(reachable, minima, 0.)
        weights = np.exp(minima[:, None] - costs)
        totals = weights.sum(axis=1)
        weights = np.divide(weights, totals[:, None], out=np.zeros_like(weights), where=totals[:, None] > 0)
        cross = covariances[:, :, :, 0].copy()
        gains = cross / variances[:, :, None]
        states += gains * innovations[:, :, None]
        covariances -= gains[:, :, :, None] * cross[:, :, None, :]
        means = np.einsum('ij,ijk->ik', weights, states)
        residuals = states - means[:, None, :]
        covs = np.einsum('ij,ijkl->ikl', weights, covariances)
        covs += np.einsum('ij,ijk,ijl->ikl', weights, residuals, residuals)
        selected_costs = np.full(len(positions), np.inf)
        selected_costs[reachable] = minima[reachable] - np.log(totals[reachable])
        normalized = selected_costs / (input_index + positions + 1.)
        winner = int(np.argmin(normalized))
        self.costs.fill(np.inf)
        self.costs[positions[reachable]] = selected_costs[reachable]
        self.states[positions[reachable]] = means[reachable]
        self.covariances[positions[reachable]] = covs[reachable]
        return int(positions[winner]), float(gamma * normalized[winner])
