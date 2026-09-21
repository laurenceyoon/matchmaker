import numpy as np

from matchmaker.dp.oltw_soft import DEFAULT_GAMMA, SoftOnlineTimeWarping
from matchmaker.prob.imm import (
    DEFAULT_OBS_VAR,
    IMMMotionModels,
    score_position_variance,
)


class IMMPathFilter:
    def __init__(self, model, variance, max_step):
        self.model = model
        self.variance = variance
        self.steps = np.arange(max_step + 1)
        self.reset()

    def reset(self, position=0):
        states, covariances, probabilities = self.model.initial_state(position)
        self.index = position
        size = len(self.variance)
        self.costs = np.full(size, np.inf)
        self.costs[position] = 0.0
        self.states = np.tile(states, (size, 1, 1))
        self.covariances = np.tile(covariances, (size, 1, 1, 1))
        self.covariances[:, :, 3, 3] = self.variance[:, None]
        self.probabilities = np.tile(probabilities, (size, 1))
        self.state = probabilities @ states

    @property
    def position(self):
        return float(self.state[0])

    def predict(self, indices):
        return self.model.predict(
            self.probabilities[indices],
            self.states[indices],
            self.covariances[indices],
            self.variance,
        )

    def restart_from(self, source, position):
        self.reset(position)
        self.states[position] = source.states[source.index]
        self.states[position, :, 0] = position
        self.covariances[position] = source.covariances[source.index]
        self.probabilities[position] = source.probabilities[source.index]
        self.state = self.probabilities[position] @ self.states[position]

    def step(self, distances, start, gamma, input_index, horizontal_weight):
        hard_min = gamma == 0
        gamma = gamma or DEFAULT_GAMMA
        if input_index == 0:
            self.costs[self.index] = float(np.sum(distances)) / gamma
            return self.index, float(distances[self.index - start])
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
        self.index = int(positions[winner])
        return self.index, float(gamma * normalized[winner])


class IMMOnlineTimeWarping(SoftOnlineTimeWarping):
    def __init__(
        self,
        *args,
        use_imm=True,
        imm_modes=("cv", "ca", "zv"),
        score_pause_gating=True,
        correlated_observation=True,
        obs_var=DEFAULT_OBS_VAR,
        **kwargs,
    ):
        self.use_imm = use_imm
        self.imm_modes = imm_modes
        self.score_pause_gating = score_pause_gating
        self.correlated_observation = correlated_observation
        self.obs_var = obs_var
        super().__init__(*args, **kwargs)

    def _create_path(self):
        if not self.use_imm:
            return super()._create_path()
        model = IMMMotionModels(
            score_part=self.score_part,
            ref_frame_to_beat=self._ref_frame_to_beat,
            obs_var=self.obs_var,
            tempo=self.tempo,
            frame_rate=self.frame_rate,
            modes=self.imm_modes,
            score_pause_gating=self.score_pause_gating,
        )
        variance = score_position_variance(
            self.score_part if self.correlated_observation else None,
            self._ref_frame_to_beat,
            self.N_ref,
        )
        return IMMPathFilter(model, variance, self.step_size)

    def get_current_position(self):
        if self.use_imm:
            return self._frame_to_beat(self.path.position)
        return super().get_current_position()
