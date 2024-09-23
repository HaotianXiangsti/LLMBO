class UD():
    # UD_id, frequency, transmission_power, cpu_cores, cpi, channel coefficients, split_point
    def __init__(self, idx, f, p, cores, cpi, h, s, k):
        self.idx = idx
        self.f = f
        self.p = p
        self.cores = cores
        self.cpi = cpi
        self.h = h
        self.s = s
        self.k = k


    def update(self, s):
        self.s = s