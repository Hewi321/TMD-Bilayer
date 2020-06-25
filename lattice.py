import itertools
import numpy as np
import math
from scipy import integrate

class lattice_3d:

    
    def __init__(self, basis, basis_top, N, a, b, theta, grid, v, symmetry, m, n, first_moire):
        self.basis = basis
        self.basis_top = basis_top
        self.N = N
        self.a = a
        self.b = b
        self.theta = theta
        self.grid = grid
        self.v = v
        self.symmetry = symmetry
        self.m = m
        self.n = n
        self.first_moire = first_moire


    def generate_rotation(self, theta):
        rot = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0,0,1]])
        return rot


    def generate_reciprocal_basis(self, basis):
        n = np.cross(basis[0], basis[1])/np.linalg.norm(np.cross(basis[0], basis[1]))
        b1 = 2*np.pi*np.cross(basis[1], n)/np.linalg.norm(np.cross(basis[0], basis[1]))
        b2 = 2*np.pi*np.cross(n, basis[0])/np.linalg.norm(np.cross(basis[0], basis[1]))
        reciprocal_basis = np.array([b1, b2])
        return reciprocal_basis


    def generate_lattice(self, basis, N):
        dimension = len(basis)
        combinations = np.array(list(itertools.product(range(-N, N+1), repeat=dimension)))
        lattice = np.matmul(basis.T, combinations.T)
        return lattice.T


    def generate_twisted_basis(self, basis, theta):
        twisted_basis = []
        
        for i in range(0,2):
            twisted_basis.append(np.dot(self.generate_rotation(theta), basis[i]))

        twisted_basis = np.array(twisted_basis)
        return twisted_basis
    
    
    def generate_moire_basis(self, basis, basis_top):
        moire0 = basis_top[0]-basis[0]
        moire1 = basis_top[1]-basis[1]
        moire_basis = np.array([moire0, moire1])
        moire_basis = self.generate_reciprocal_basis(moire_basis)
        return moire_basis
    

    def generate_first_reciprocal_moire(self, basis, basis_top):
        moire_lattice = []
        moire_norm = []
        lattice = self.generate_lattice(basis, 1)
        twisted_lattice = self.generate_lattice(basis_top, 1)
        
        for i in range(0,len(lattice)):
            moire_lattice.append(twisted_lattice[i]-lattice[i])
            moire_norm.append(np.linalg.norm(moire_lattice[i]))
            
        moire_lattice = np.array(moire_lattice)
        moire_norm = np.array(moire_norm)
        position_min = moire_norm.argmin()
        moire_lattice = np.delete(moire_lattice, (position_min), axis=0) 
        moire_norm = np.delete(moire_norm, (position_min), axis=0) 
        position_max = moire_norm.argmax()
        moire_lattice = np.delete(moire_lattice, (position_max), axis=0)
        moire_norm = np.delete(moire_norm, (position_max), axis=0)
        position_max = moire_norm.argmax()
        moire_lattice = np.delete(moire_lattice, (position_max), axis=0)
        moire_norm = np.delete(moire_norm, (position_max), axis=0)
        return moire_lattice

      
    def generate_monkhorst_raw(self, basis, N):
        monkhorst_raw = []
        monkhorst_raw_norm = []
        
        for i in range(1, N+1):
            for j in range(1, N+1):
                monkhorst_raw.append((-0.5+i/N)*basis[0]+(-0.5+j/N)*basis[1])
                monkhorst_raw_norm.append(np.linalg.norm(monkhorst_raw))
        
        monkhorst_raw = np.array(monkhorst_raw)
        monkhorst_raw_norm = np.array(monkhorst_raw_norm)
        return monkhorst_raw, monkhorst_raw_norm


    def generate_monkhorst(self, basis, N):
        monkhorst_raw_result = self.generate_monkhorst_raw(basis, N)
        monkhorst_raw = monkhorst_raw_result[0]
        monkhorst_norm = monkhorst_raw[1]
        monkhorst = []
        
        if N % 2:
            monkhorst = monkhorst_raw
                      
        else:
            reference_value = np.argmin(monkhorst_norm)
            for i in range(0, len(monkhorst_raw)):
                monkhorst.append(monkhorst_raw[i]-monkhorst_raw[reference_value])
      
        monkhorst = np.array(np.round(monkhorst,3))
        return monkhorst_raw


    def generate_potential(self, v, first_moire, theta, grid, symmetry):
        V_const = np.array([-v*np.exp(1j*symmetry), v*np.exp(1j*symmetry), v*np.exp(1j*symmetry), -v*np.exp(1j*symmetry), -v*np.exp(1j*symmetry), v*np.exp(1j*symmetry)])
        V = np.sum(np.real(np.exp(1j*(np.tensordot(first_moire[:,0], grid[0], axes=0)+np.tensordot(first_moire[:,1], grid[1], axes=0)))*V_const[:, None, None]), axis=0)
        return V

    
    def generate_reciprocal_potential(self, v, first_moire, grid, symmetry):
        int_radius = np.linalg.norm(first_moire[0])
        mbz_area = 0.5*3*np.sqrt(3)*np.linalg.norm(first_moire[0])**2
        V_const = np.array([-v*np.exp(1j*symmetry), v*np.exp(1j*symmetry), v*np.exp(1j*symmetry), -v*np.exp(1j*symmetry), -v*np.exp(1j*symmetry), v*np.exp(1j*symmetry)])
        func = lambda r, phi : (1/mbz_area)*r*np.exp(-1j*(np.tensordot(grid[0], np.array([r*np.cos(phi), r*np.sin(phi)]), axes=0)+np.tensordot(grid[1], np.array([r*np.cos(phi), r*np.sin(phi)]), axes=0)))*np.sum(V_const[:, None, None]*np.exp(1j*np.dot(first_moire[:, None, None], np.array([r*np.cos(phi), r*np.sin(phi)]))))
        V_reciprocal = integrate.dblquad(func, 0, int_radius, 0, 2*np.pi)
        return func