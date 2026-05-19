import numpy as np

DEFAULT_SEED: list[int] = list(map(ord, "Home is where the other members of group 3 are"))

class Simulator:
    def __init__(self, model: dict):
        self.state_dim: int = model["state_dim"]
        self.input_dim: int = model["input_dim"]
        self.obs_dim: int = model["obs_dim"]

        if (A := model.get("A", np.zeros(0))).shape != (self.state_dim, self.state_dim):
            raise ValueError("A must match dimensions of the state")
        if (B := model.get("B", np.zeros(0))).shape != (self.state_dim, self.input_dim):
            raise ValueError("B must match the state and input dimensions")
        if (Q := model.get("Q", np.zeros(0))).shape != (self.state_dim, self.state_dim):
            raise ValueError("Q must match dimensions of the state")
        if (C := model.get("C", np.zeros(0))).shape != (self.obs_dim, self.state_dim):
            raise ValueError("C must match the state and observation dimensions")
        if (R := model.get("R", np.zeros(0))).shape != (self.obs_dim, self.obs_dim):
            raise ValueError("R must match dimensions of the observation")
        
        self.A: np.ndarray = A
        self.B: np.ndarray = B
        self.Q: np.ndarray = Q
        self.C: np.ndarray = C
        self.R: np.ndarray = R

        self.generator = np.random.default_rng(seed=model.get("seed", DEFAULT_SEED))

    def run(self, initial_state: np.ndarray, control_inputs: np.ndarray, time_steps: int, trials=1) -> tuple[np.ndarray, np.ndarray]:
        if initial_state.shape != (self.state_dim,):
            raise ValueError(f"Initial state does not match the specified state dimension ({initial_state.shape} vs {self.state_dim})")
        if control_inputs.shape != (time_steps, self.input_dim):
            raise ValueError(f"Control input does not match the specified time steps and input dimension ({control_inputs.shape} vs {(time_steps, self.input_dim)})")

        states: np.ndarray = np.zeros((trials, time_steps, self.state_dim))
        observations: np.ndarray = np.zeros((trials, time_steps, self.obs_dim))

        states[:, 0] = initial_state
        observations[:, 0] = self.C @ initial_state

        for trial in range(trials):
            for t in range(1, time_steps):
                state_noise: np.ndarray = self.generator.multivariate_normal(
                        np.zeros(self.state_dim),
                        self.Q
                    )
                
                obs_noise: np.ndarray = self.generator.multivariate_normal(
                        np.zeros(self.obs_dim),
                        self.R
                    )
                
                states[trial, t] = self.A @ states[trial, t - 1] + self.B @ control_inputs[t - 1] + state_noise
                observations[trial, t] = self.C @ states[trial, t] + obs_noise

        return states, observations

    def test(self):
        print("Simulator test method called.")
