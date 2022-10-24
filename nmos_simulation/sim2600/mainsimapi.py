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

import time
import params

import imagePIL
from sim2600Console import Sim2600Console
import sim6502
import simTIA

class MainSim:
    """
    This is a sim runner designed to be invoked from an
    outer experiment runner. We pass in the romfile, 
    and manually advance the clock ourselves

    Only render with images, no OpenGL 
    """
    def __init__(self, romFile, imgdir,
                 sim6502factory = sim6502.MySim6502):
        self.imagePIL = imagePIL.getInterface(imgdir)
        self.elapsedHalfClocks = 0

        # The console simulator ties together a simulation
        # of the 6502 chip, a simulation of the TIA chip, 
        # an emulation of the 6532 RIOT (RAM, I/O, Timer), and
        # a cartridge ROM file holding the program instructions.
        #
        #self.sim = Sim2600Console(params.romFile)
        self.sim = Sim2600Console(romFile, sim6502factory, simTIA.MySimTIA)

        # For measuring how fast the simulation is running
        self.lastUpdateTimeSec = None

                
    def updateSimOneFrame(self, logger=None, eventLogger = None):
        tia = self.sim.simTIA
        pixels = []
        
        if eventLogger is None:
            def eventLogger(*args):
                pass

        i = 0
        while i < params.numTIAHalfClocksPerRender:
            self.sim.advanceOneHalfClock()

            # Get pixel color when TIA clock (~3mHz) is low
            if tia.isLow(tia.padIndCLK0):

                restartImage = False
                if self.sim.simTIA.isHigh(self.sim.simTIA.vsync):
                    print('VSYNC high at TIA halfclock %d'%(tia.halfClkCount))
                    eventLogger('VSYNC high', tia.halfClkCount, 
                                self.sim.sim6507.halfClkCount)
                    restartImage = True

                # grayscale
                #lum = self.simTIA.get3BitLuminance() << 5
                #rgba = (lum << 24) | (lum << 16) | (lum << 8) | 0xFF

                # color
                rgba = self.sim.simTIA.getColorRGBA8()

                if self.imagePIL != None:
                    if restartImage == True:
                        self.imagePIL.restartImage()
                        eventLogger('restartImage', tia.halfClkCount, 
                                    self.sim.sim6507.halfClkCount)

                    self.imagePIL.setNextPixel(rgba)
                    eventLogger('setNextPixel', tia.halfClkCount, 
                                self.sim.sim6507.halfClkCount, rgba)

                if self.sim.simTIA.isHigh(self.sim.simTIA.vblank):
                    print('VBLANK at TIA halfclock %d'%(tia.halfClkCount))
                    eventLogger('VBLANK', tia.halfClkCount, 
                                self.sim.sim6507.halfClkCount)

                #cpuStr = self.sim6502.getStateStr1()
                #tiaStr = self.simTIA.getTIAStateStr1()
                #print(cpuStr + '   ' + tiaStr)
            if logger is not None:
                logger(self.sim, i)
            i += 1

        if self.lastUpdateTimeSec != None:
            elapsedSec = time.time() - self.lastUpdateTimeSec
            secPerSimClock = 2.0 * elapsedSec / params.numTIAHalfClocksPerRender
            totalWires = self.sim.sim6507.numWiresRecalculated + \
                         self.sim.simTIA.numWiresRecalculated
            wiresPerClock = 2.0 * totalWires / params.numTIAHalfClocksPerRender
            print('                                          ' + 
                  '%d wires/clk,  %g msec/clk'%
                  (wiresPerClock, secPerSimClock * 1000))
            self.sim.sim6507.numWiresRecalculated = 0
            self.sim.simTIA.numWiresRecalculated = 0
        self.lastUpdateTimeSec = time.time()

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


