
# taken from https://github.com/enthought/mayavi/blob/master/mayavi/tools/server.py
# can I use this?

import wx
import numpy as np
import time

from optrace.tracer import *

# Install wxreactor; must be done before the reactor is imported below.
from twisted.internet import wxreactor
wxreactor.install()

# The usual twisted imports.
from twisted.internet.protocol import Protocol, DatagramProtocol, Factory
from twisted.internet import reactor
from twisted.python import log


class M2TCP(Protocol):

    maxConnect = 1
    history = b""

    def connectionMade(self):
        log.msg('TCP Connection Made')
        self.factory.numConnect += 1
        if self.factory.numConnect > self.maxConnect:
            self.transport.write(b"Server has already reached maximum number of connections.\n")
            self.transport.loseConnection()

    def connectionLost(self, reason):
        log.msg('TCP Connection Lost')
        self.factory.numConnect -= 1

    def dataReceived(self, data):
        """Given a line of data, simply execs it to do whatever."""

        c = data.strip()

        if len(c) > 0:

            try:
                log.msg('Received command:', c)
                self.history += c + b"\n"
       
                gui = self.factory.GUI

                if not gui.busy:

                    hs = gui.Raytracer.PropertySnapshot()
                    exec(c, locals() | self.factory.dict_, globals())

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

    def printHistory(self):
        log.msg(b"\n\n" + self.history)


def serve_tcp(GUI, port=8007, logto=None, max_connect=1, dict_=dict()):

    # Setup the factory with the right attributes.
    factory = Factory()
    factory.protocol = M2TCP
    factory.maxConnect = max_connect
    factory.numConnect = 0

    factory.GUI = GUI
    factory.dict_ = dict_

    log.msg('Serving Mayavi2 TCP server on port', port)

    # Register the running wxApp.
    reactor.registerWxApp(wx.GetApp())

    # Listen on port 8007 using above protocol.
    reactor.listenTCP(port, factory)

    # Run the server + app.  This will block.
    reactor.run()


