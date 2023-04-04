import os
import numpy as np
from sim2600 import sim2600Console
from sim2600 import params, sim6502, simTIA


if __name__ == "__main__":
    num_iterations = 512
    step_limit = 400
    intervention_step = 385
    transistor_idx = 1
    transistor_history = []

    # without intervention
    sim = sim2600Console.Sim2600Console(params.ROMS_DONKEY_KONG, simTIAfactory=simTIA.MySimTIA,
                                        sim6502factory=sim6502.MySim6502)

    for step in range(num_iterations):
        # Run the simulation for one iteration
        if step == intervention_step:
            sim.sim6507.setRecording(True)

        sim.advanceOneHalfClock()

        # Record the transistor states, only consider the next half-clock
        if step == intervention_step:
            state = sim.sim6507.getFullTransistorState()
            state = np.concatenate((state[:, :-1], np.tile(state[:, -2].reshape(-1, 1), step_limit - state.shape[1] + 1)), axis=1)
            last_state = state[transistor_idx, -1]

        if step == intervention_step + 1:
            state = sim.sim6507.getFullTransistorState()
            transistor_history = np.concatenate((state[:, :-1], np.tile(state[:, -2].reshape(-1, 1), step_limit - state.shape[1] + 1)), axis=1)
            current_state = transistor_history[transistor_idx, 0]
            break

    # with intervention
    sim = sim2600Console.Sim2600Console(params.ROMS_DONKEY_KONG, simTIAfactory=simTIA.MySimTIA,
                                                     sim6502factory=sim6502.MySim6502)

    for step in range(num_iterations):
        # Run the simulation for one iteration
        if step == intervention_step + 1:
            sim.sim6507.setRecording(True)

        sim.advanceOneHalfClock()

        # Record the transistor states, only consider the next half-clock
        if step == intervention_step + 1:
            state = sim.sim6507.getFullTransistorState()
            # pad to the steplimit length
            state = np.concatenate((state[:, :-1], np.tile(state[:, -2].reshape(-1, 1), step_limit - state.shape[1] + 1)), axis=1)
            intervention_transistor_history = state
            break

        # Intervene on a transistor if needed
        if step == intervention_step:
            print('original state: ', current_state)
            intervention_state = 0 if current_state == 1 else 1
            sim.sim6507.intervenTransistor(transistor_idx, intervention_state)
            print('intervention state: ', sim.sim6507.getTransistorState()[transistor_idx])

    # compare the transistor states with and without intervention
    comparison_step = intervention_step+1
    print(intervention_transistor_history)
    print('intervention result and regular result are the same: ',
          np.all(intervention_transistor_history == transistor_history))
    print('affected transistors: ',
          set(np.where(intervention_transistor_history != transistor_history)[0].tolist()))

