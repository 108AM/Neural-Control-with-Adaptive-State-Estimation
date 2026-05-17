class Simulator:
    def __init__(self, model):
        self.A = model["A"]
        self.B = model["B"]
        self.C = model["C"]

    def run(self, initial_state, control_inputs, time_steps):
        states = [initial_state]
        for t in range(time_steps):
            current_state = states[-1]
            control_input = control_inputs[t]
            next_state = self.A @ current_state + self.B @ control_input
            states.append(next_state)
        return states

    def test(self):
        print("Simulator test method called.")
