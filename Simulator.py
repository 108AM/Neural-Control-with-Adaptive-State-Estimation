class Simulator():
    def __init__(self, model, dt=0.01):
        self.model = model
        self.dt = dt

    def run(self, initial_state, control_inputs, time_steps):
        states = [initial_state]
        for t in range(time_steps):
            current_state = states[-1]
            control_input = control_inputs[t]
            next_state = self.model(current_state, control_input) * self.dt + current_state
            states.append(next_state)
        return states

    def test(self):
        print("Simulator test method called.")