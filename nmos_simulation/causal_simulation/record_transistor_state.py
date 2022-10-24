import numpy as np
from sim2600 import sim2600Console
from sim2600 import params, sim6502, simTIA


def transistor_record_hr(lesion=None, halfclk=-1, rom=params.ROMS_DONKEY_KONG, iteration=1000, tidx=-1):
    """
    Record the transistor state for a given rom and iteration in higher temporal resolution (within the half-clock)
    """
    if lesion is None:
        # Regular simulation and not lesioning
        tidx = -1
        s1 = lambda x: sim2600Console.Sim2600Console(x, simTIAfactory=simTIA.MySimTIA,
                                                     sim6502factory=sim6502.MySim6502,
                                                     tidx_lesion=tidx,)
    else:
        # Simulation with lesioning a transistor
        s1 = lambda x: sim2600Console.Sim2600Console(x, simTIAfactory=simTIA.MySimTIA,
                                                     sim6502factory=sim6502.MySim6502,
                                                     lesion=lesion,
                                                     tidx_lesion=tidx,)

    s1 = s1(rom)
    t1_state = []
    halfclk = [halfclk] if halfclk != -1 else range(iteration)
    s1.sim6507.resetFullTransistorState()

    # np.testing.assert_array_equal(t1_init_state, t2_init_state)
    print("-" * 50)
    for i in range(iteration):
        # turn on the full resolution recording for nmos6502 at specific clock cycle
        if i in halfclk:
            s1.sim6507.setRecording(True)
        else:
            s1.sim6507.setRecording(False)

        # half-clock step the 6502
        s1.advanceOneHalfClock()

        #  record the transistor state
        if i in halfclk:
            t1_state.append(s1.sim6507.getFullTransistorState())

    # return transistor state
    t1_state = t1_state[0] if len(t1_state) == 1 else np.concatenate(t1_state, axis=1)
    return t1_state


def original_measure_hr(rom=params.ROMS_DONKEY_KONG, iteration=1000):
    t_org = transistor_record_hr(rom=rom, iteration=iteration)
    return t_org


def single_leision_measure_v2(tidx=-1, halfclk=-1, lesion="High", rom=params.ROMS_DONKEY_KONG, iteration=1000):
    # data0 = -1 * np.ones((3510, 50000), dtype=np.int8)
    t_org = transistor_record_hr(lesion, halfclk, rom, iteration=iteration, tidx=tidx)
    return t_org