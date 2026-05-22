import numpy as np

from l4b import Simulator, analyse_controllability

test = Simulator(
    model=(
        {
            "state_dim": 2,
            "input_dim": 2,
            "obs_dim": 2,

            "A": np.array([
                    [0.9, 0.1],
                    [0.2, 0.8]
                ]),
            "B": np.array([
                    [0.1, 0.3],
                    [0.3, 0.1]
                ]),
            "Q": np.array([
                    [1.0, 0.0],
                    [1.0, 0.0]
                ]),
            
            "C": np.array([
                    [1.0, 0.0],
                    [0.0, 1.0]
                ]),
            "R": np.array([
                    [1.0, 0.0],
                    [0.0, 1.0]
                ])
        }
    )
)

states, observations = test.run(
    initial_state=np.array([1.0, 1.0]), control_inputs=np.zeros((50, 2)), time_steps=50, trials=2
)

W_c = test.controllability_gramian(10)
print(analyse_controllability(W_c))

# illustrator = ill.Illustrator(observations)
# illustrator.plot_all()
