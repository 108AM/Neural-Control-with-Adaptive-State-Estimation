import numpy as np

DEFAULT_SEED: list[int] = list(
    map(ord, "Home is where the other members of group 3 are")
)


class Simulator:
    def __init__(self, model: dict):
        self.state_dim: int = model["state_dim"]
        self.input_dim: int = model["input_dim"]
        self.obs_dim: int = model["obs_dim"]

        def _validate(name: str, shape: tuple[int, int]) -> np.ndarray:
            M = np.asarray(model.get(name, np.zeros(0)))
            if M.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {M.shape}")
            return M

        self.A: np.ndarray = _validate("A", (self.state_dim, self.state_dim))
        self.B: np.ndarray = _validate("B", (self.state_dim, self.input_dim))
        self.Q: np.ndarray = _validate("Q", (self.state_dim, self.state_dim))
        self.C: np.ndarray = _validate("C", (self.obs_dim, self.state_dim))
        self.R: np.ndarray = _validate("R", (self.obs_dim, self.obs_dim))

        self.generator = np.random.default_rng(seed=model.get("seed", DEFAULT_SEED))

    def run(
        self,
        initial_state: np.ndarray,
        control_inputs: np.ndarray,
        time_steps: int,
        trials=1,
    ) -> tuple[np.ndarray, np.ndarray]:
        if initial_state.shape != (self.state_dim,):
            raise ValueError(
                f"Initial state does not match the specified state dimension ({initial_state.shape} vs {self.state_dim})"
            )
        if control_inputs.shape != (time_steps, self.input_dim):
            raise ValueError(
                f"Control input does not match the specified time steps and input dimension ({control_inputs.shape} vs {(time_steps, self.input_dim)})"
            )

        states: np.ndarray = np.zeros((trials, time_steps, self.state_dim))
        observations: np.ndarray = np.zeros((trials, time_steps, self.obs_dim))

        states[:, 0] = initial_state
        observations[:, 0] = self.C @ initial_state

        for trial in range(trials):
            for t in range(1, time_steps):
                state_noise: np.ndarray = self.generator.multivariate_normal(
                    np.zeros(self.state_dim), self.Q
                )

                obs_noise: np.ndarray = self.generator.multivariate_normal(
                    np.zeros(self.obs_dim), self.R
                )

                states[trial, t] = (
                    self.A @ states[trial, t - 1]
                    + self.B @ control_inputs[t - 1]
                    + state_noise
                )
                observations[trial, t] = self.C @ states[trial, t] + obs_noise

        return states, observations
