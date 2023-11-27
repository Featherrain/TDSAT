from models.TDSAT.TDSAT import TDSAT


def tdsat():
    net = TDSAT(1, 16, 5, [1, 3])
    net.use_2dconv = False
    net.bandwise = False
    return net
