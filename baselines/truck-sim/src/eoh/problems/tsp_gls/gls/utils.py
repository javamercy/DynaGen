from numba import jit


@jit(nopython=True)
def tour_cost_2End(dis_m, tour2End):
    c=0
    s = 0
    e = tour2End[0,1]
    for i in range(tour2End.shape[0]):
        c += dis_m[s,e]
        s = e
        e = tour2End[s,1]
    return c
