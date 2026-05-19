import numpy as np
import simulator as sim
import illustrator as ill

test = sim.Simulator(
    model=(
        {
            "state_dim": 2,
            "input_dim": 2,
            "obs_dim": 1,

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
                    [1.0, 0.0]
                ]),
            "R": np.array([[0]]) # No noise for now!
        }
    )
)

states, observations = test.run(
    initial_state=np.array([1.0, 1.0]), control_inputs=np.zeros((50, 2)), time_steps=50, trials=2
)

illustrator = ill.Illustrator(observations)
illustrator.plot_all()
