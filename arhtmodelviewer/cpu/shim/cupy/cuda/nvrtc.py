"""cupy.cuda.nvrtc on the CPU: the version ArWen's identity checks read."""


class NVRTCError(RuntimeError):
    pass


def getVersion():
    return (12, 9)
