
import wx
import numpy as np


from optrace.tracer import Lens, Filter, Aperture, RaySource, Detector, Raytracer, RImage,\
                           Surface, SurfaceFunction, Spectrum, TransmissionSpectrum, LightSpectrum, RefractionIndex

from twisted.internet import wxreactor
wxreactor.install()

from twisted.internet.protocol import Protocol, DatagramProtocol, Factory
from twisted.internet import reactor
from twisted.python import log


class OTTCP(Protocol):

    def __init__(self):
        self.maxConnect = 1
        self.history = b""

    def connectionMade(self):
        """do when connection is established"""

        log.msg('TCP Connection Made')
        self.factory.numConnect += 1

        if self.factory.numConnect > self.maxConnect:
            self.transport.write(b"Server has already reached maximum number of connections.\n")
            self.transport.loseConnection()

    def connectionLost(self, reason):
        """do when connection is lost"""
        log.msg('TCP Connection Lost')
        self.factory.numConnect -= 1

    def printHistory(self):
        """print all sent commands"""
        log.msg(b"\n\n" + self.history)

    def dataReceived(self, data):
        """exec received data."""

        if (cmd := data.strip()):

            try:
                log.msg('Received command:', cmd)
                self.history += cmd + b"\n"
       
                gui = self.factory.GUI
                printHistory = self.printHistory

                if not gui.busy:

                    hs = gui.Raytracer.PropertySnapshot()
                    exec(cmd, locals() | self.factory.dict_, globals())

                    hs2 = gui.Raytracer.PropertySnapshot()
                    cmp = gui.Raytracer.comparePropertySnapshot(hs, hs2)
                    gui.replot(cmp)
                else:
                    log.msg("Currently busy, not starting a new action.")

            except SystemExit:
                exit()

            except Exception:
                log.err()
        else:
            log.msg('Received empty command')


def serve_tcp(GUI, port=8007, max_connect=1, dict_=dict()):

    # Setup the factory with the right attributes.
    factory = Factory()
    factory.protocol = OTTCP
    factory.maxConnect = max_connect
    factory.numConnect = 0

    factory.GUI = GUI
    factory.dict_ = dict_

    log.msg('Serving Optrace TCP server on port', port)

    # Register the running wxApp.
    reactor.registerWxApp(wx.GetApp())

    # Listen on port 8007 using above protocol.
    reactor.listenTCP(port, factory)

    # Run the server + app.  This will block.
    reactor.run()

