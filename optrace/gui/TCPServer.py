import wx  # tcp protocol is run inside wx app

# configures the twisted mainloop to be run inside the wxPython mainloop. 
from twisted.internet import wxreactor
wxreactor.install()

from twisted.internet import reactor  # provides networking
from twisted.internet.protocol import Protocol, DatagramProtocol, Factory  # needed for tcp protocol
from twisted.python import log  # logging
import twisted.python.failure  # for Failure type


TCP_port = 8007
"""port for the TCP server"""


class OTTCP(Protocol):

    def __init__(self) -> None:
        self.maxConnect = 1  # refuse connections after this number
        self.history = b""  # cmd history

    def connectionMade(self) -> None:
        """do when connection is established"""

        log.msg('TCP Connection Made')
        self.factory.numConnect += 1

        if self.factory.numConnect > self.maxConnect:
            self.transport.write(b"Server has already reached maximum number of connections.\n")
            self.transport.loseConnection()

    def connectionLost(self, reason: twisted.python.failure.Failure) -> None:
        """
        do when connection is lost

        :param reason: Failure object
        """
        log.msg(f"TCP Connection Lost. Reason: {reason}")
        self.factory.numConnect -= 1

    def printHistory(self) -> None:
        """print all sent commands"""
        log.msg(b"\n\n" + self.history)

    def dataReceived(self, data: bytes) -> None:
        """
        exec received data in main thread

        :param data: received byte string
        """

        if cmd := data.strip():

            try:
                log.msg('Received command:', cmd)
                self.history += cmd + b"\n"

                gui = self.factory.GUI
                printHistory = self.printHistory

                # TODO can we do this differently?
                # while being thread safe, correctly detecting and not having to wait unnecessarily
                if not gui.busy:

                    gui._process_events()  # do outstanding UI events
                    hs = gui.Raytracer.PropertySnapshot()
                    # Note: this is run in the main thread
                    exec(cmd, locals() | self.factory.dict_, globals())

                    hs2 = gui.Raytracer.PropertySnapshot()
                    cmp = gui.Raytracer.comparePropertySnapshot(hs, hs2)
                    gui.replot(cmp)
                else:
                    log.msg("Currently busy, not starting a new action.")

            # compact error message. command is already printed in the try-block
            except SyntaxError:
                log.err('Syntax error in command')
                # log.err(cmd)

            # only exit connection
            except SystemExit:
                self.transport.loseConnection()
                self.factory.numConnect -= 1

            except Exception:
                log.err()
        else:
            log.msg('Received empty command')


def serve_tcp(GUI:          'TraceGUI', 
              port:         int = TCP_port, 
              max_connect:  int = 1, 
              dict_:        dict = None)\
        -> None:
    """

    :param GUI:
    :param port:
    :param max_connect:
    :param dict_:
    """

    # Setup the factory with the right attributes.
    factory = Factory()
    factory.protocol = OTTCP
    factory.maxConnect = max_connect
    factory.numConnect = 0

    factory.GUI = GUI
    factory.dict_ = dict_ if dict_ is not None else dict()

    log.msg('Serving Optrace TCP server on port', port)

    # Register the running wxApp.
    reactor.registerWxApp(wx.GetApp())

    # Listen on port 8007 using above protocol.
    reactor.listenTCP(port, factory)

    # Run the server + app.  This will block.
    reactor.run()

