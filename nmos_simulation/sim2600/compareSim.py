#!/usr/bin/python
# Copyright (c) 2014 Greg James, Visual6502.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import params

from sim2600Console import Sim2600Console

class MainSim:
    def __init__(self):
        self.elapsedHalfClocks = 0

        self.sim = Sim2600Console(params.romFile)

        # For measuring how fast the simulation is running
        self.lastUpdateTimeSec = None

    def rumsim(self, halfClocks=100):
        for i in range(halfClocks):
            self.sim.advanceOneHalfClock()

        # if self.lastUpdateTimeSec != None:
        #     elapsedSec = time.time() - self.lastUpdateTimeSec
        #     secPerSimClock = 2.0 * elapsedSec / params.numTIAHalfClocksPerRender
        #     totalWires = self.sim.sim6507.numWiresRecalculated + \
        #                  self.sim.simTIA.numWiresRecalculated
        #     wiresPerClock = 2.0 * totalWires / params.numTIAHalfClocksPerRender
        #     print('                                          ' + 
        #           '%d wires/clk,  %g msec/clk'%
        #           (wiresPerClock, secPerSimClock * 1000))
        #     self.sim.sim6507.numWiresRecalculated = 0
        #     self.sim.simTIA.numWiresRecalculated = 0
        # self.lastUpdateTimeSec = time.time()

    def getStateStr(self):
        cpu = self.sim.sim6507
        tia = self.sim.simTIA
        sstr =  'CLKS %d%d'%(cpu.isHighWN('CLK0'), tia.isHighWN('CLK0')) + ' '
        sstr += 'RS,RDY %d%d'%(cpu.isHighWN('RES'), cpu.isHighWN('RDY')) + ' '
        sstr += 'ADDR 0x%4.4X  DB 0x%2.2X  '% \
                (self.sim.sim6507.getAddressBusValue(),
                 self.sim.sim6507.getDataBusValue())
        sstr += self.simTIA.getTIAStateStr1()
        return sstr


if __name__ == '__main__':
    printStartupMsg()
    MainSim()
    print('Exiting mainSim.py')
