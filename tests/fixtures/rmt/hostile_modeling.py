import base64 as b64
import os as operating_system
import subprocess as process
from os import system as run_system
from pty import spawn


run_system("id")
operating_system.popen("id")
process.Popen(["id"])
spawn("/bin/sh")


@operating_system.system
def decorated():
    pass


class ImportSideEffect:
    operating_system.execv("/bin/sh", ["/bin/sh"])


def configured(value=eval("40 + 2")):
    return value


exec(b64.b64decode("cHJpbnQoJ3gnKQ=="))
