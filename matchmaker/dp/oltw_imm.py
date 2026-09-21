import numpy as np

from matchmaker.dp.oltw_soft import SoftOnlineTimeWarping
from matchmaker.prob.imm import ScoreInformedIMM, score_position_variance


class IMMPathFilter:
    def __init__(self, model, variance, max_step):
        self.model = model
        self.variance = variance
        self.steps = np.arange(max_step + 1)
        self.reset()

    def reset(self):
        self.model.reset()
        size = len(self.variance)
        self.costs = np.full(size, np.inf)
        self.costs[0] = 0.0
        self.states = np.tile(self.model.states, (size, 1, 1))
        self.covariances = np.tile(self.model.P_matrices, (size, 1, 1, 1))
        self.covariances[:, :, 3, 3] = self.variance[:, None]
        self.probabilities = np.tile(self.model.mu, (size, 1))
        self.state = self.model.state.copy()

    @property
    def position(self):
        return float(self.state[0])

    def predict(self, indices):
        probabilities = self.probabilities[indices]
        states, covariance = self.states[indices], self.covariances[indices]
        means = np.einsum("am,amd->ad", probabilities, states)
        beats = means[:, 0]
        if self.model.r2b is not None:
            beats = np.interp(beats, np.arange(len(self.model.r2b)), self.model.r2b)
        allowed = np.full(len(indices), not self.model.score_pause_gating)
        for start, end in self.model.pause_ranges:
            allowed |= (beats >= start) & (beats < end)
        transition = np.where(
            allowed[:, None, None], self.model.M_pause, self.model.M_play
        )
        joint = probabilities[:, :, None] * transition
        priors = joint.sum(axis=1)
        mixing = np.divide(
            joint,
            priors[:, None, :],
            out=np.zeros_like(joint),
            where=priors[:, None, :] > 0,
        )
        mixed = np.einsum("aij,aid->ajd", mixing, states)
        residuals = states[:, :, None, :] - mixed[:, None, :, :]
        covariance = np.einsum("aij,aikl->ajkl", mixing, covariance)
        covariance += np.einsum("aij,aijk,aijl->ajkl", mixing, residuals, residuals)
        frames = np.clip(np.rint(means[:, 0]).astype(int), 0, len(self.variance) - 1)
        variance = self.variance[frames]
        duration = np.sqrt(12 * variance)
        correlation = np.exp(
            np.divide(
                -1.0, duration, out=np.full_like(duration, -np.inf), where=duration > 0
            )
        )
        dynamics = np.tile(self.model.F, (len(indices), 1, 1, 1))
        noise = np.tile(self.model.Q, (len(indices), 1, 1, 1))
        dynamics[:, :, 3, 3] = correlation[:, None]
        noise[:, :, 3, 3] = (variance * (1 - correlation**2))[:, None]
        predicted = np.einsum("amij,amj->ami", dynamics, mixed)
        covariance = dynamics @ covariance @ dynamics.swapaxes(-1, -2) + noise
        return priors, predicted, covariance

    def step(
        self, distances, start, gamma, input_index, horizontal_weight, hard_min=False
    ):
        if input_index == 0:
            self.costs[0] = float(np.sum(distances)) / gamma
            return 0, float(distances[0])
        positions = np.arange(start, start + len(distances))
        ancestors = positions[:, None] - self.steps
        valid = ancestors >= start
        ancestors = np.maximum(ancestors, start)
        priors, states, covariances = self.predict(positions)
        priors = priors[ancestors - start]
        states = states[ancestors - start]
        covariances = covariances[ancestors - start]
        innovations = positions[:, None, None] - states @ self.model.H
        cross = covariances @ self.model.H
        variances = cross @ self.model.H + self.model.R
        log_priors = np.log(priors, out=np.full_like(priors, -np.inf), where=priors > 0)
        transitions = (
            0.5 * (innovations**2 / variances + np.log(2 * np.pi * variances))
            - log_priors
        )
        cumulative = np.r_[0.0, np.cumsum(distances)]
        left = np.maximum(positions[:, None] - self.steps + 1 - start, 0)
        emission = cumulative[np.arange(len(positions))[:, None] + 1] - cumulative[left]
        emission[:, 0] = horizontal_weight * distances
        costs = self.costs[ancestors, None] + transitions + emission[:, :, None] / gamma
        costs = np.where(valid[:, :, None], costs, np.inf)
        minima = costs.min(axis=(1, 2))
        reachable = np.isfinite(minima)
        minima = np.where(reachable, minima, 0.0)
        weights = np.exp(minima[:, None, None] - costs)
        if hard_min:
            best = weights.sum(axis=2).argmax(axis=1)
            weights *= self.steps[None, :, None] == best[:, None, None]
        totals = weights.sum(axis=(1, 2))
        mode_totals = weights.sum(axis=1)
        probabilities = np.divide(
            mode_totals,
            totals[:, None],
            out=np.zeros_like(mode_totals),
            where=totals[:, None] > 0,
        )
        mixing = np.divide(
            weights,
            mode_totals[:, None, :],
            out=np.zeros_like(weights),
            where=mode_totals[:, None, :] > 0,
        )
        gains = cross / variances[..., None]
        states += gains * innovations[..., None]
        covariances -= gains[..., :, None] * cross[..., None, :]
        means = np.einsum("asm,asmd->amd", mixing, states)
        residuals = states - means[:, None, :, :]
        covariance = np.einsum("asm,asmij->amij", mixing, covariances)
        covariance += np.einsum("asm,asmi,asmj->amij", mixing, residuals, residuals)
        selected_costs = np.full(len(positions), np.inf)
        selected_costs[reachable] = minima[reachable] - np.log(totals[reachable])
        normalized = selected_costs / (input_index + positions + 1.0)
        winner = int(np.argmin(normalized))
        self.costs.fill(np.inf)
        self.costs[positions] = selected_costs
        self.states[positions] = means
        self.covariances[positions] = covariance
        self.probabilities[positions] = probabilities
        self.state = probabilities[winner] @ means[winner]
        return int(positions[winner]), float(gamma * normalized[winner])


class IMMOnlineTimeWarping(SoftOnlineTimeWarping):
    def __init__(
        self,
        *args,
        imm_modes=("cv", "ca", "zv"),
        score_pause_gating=True,
        correlated_observation=True,
        **kwargs,
    ):
        self.path_filter = None
        super().__init__(
            *args, use_imm=False, path_tempo=False, filter_output=False, **kwargs
        )
        model = ScoreInformedIMM(
            score_part=self.score_part,
            ref_frame_to_beat=self._ref_frame_to_beat,
            obs_var=self._obs_var,
            tempo=self.tempo,
            frame_rate=self.frame_rate,
            modes=imm_modes,
            score_pause_gating=score_pause_gating,
        )
        variance = score_position_variance(
            self.score_part if correlated_observation else None,
            self._ref_frame_to_beat,
            self.N_ref,
        )
        self.path_filter = IMMPathFilter(model, variance, self.step_size)
        self.reset()

    def reset(self):
        super().reset()
        if self.path_filter is not None:
            self.path_filter.reset()
            self.path_lattice = self.path_filter
            self.global_cost_matrix = None

    def get_current_position(self):
        return self._frame_to_beat(self.path_filter.position)
