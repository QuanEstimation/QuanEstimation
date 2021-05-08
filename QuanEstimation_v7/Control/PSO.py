import numpy as np


class PSO(object):
    def __init__(self, rho0, particle_num, particle_dim, c0=1.0, c1=2.0, c2=2.0):
        """
        --------
        inputs
        --------
        
        """
        self.rho0 = rho0
        self.particle_num = particle_num
        self.particle_dim = particle_dim
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        
    def swarm_origin(self):
        
        for pi in range(self.particle_num):
            rho
        