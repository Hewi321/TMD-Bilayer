import numpy as np
import math
import matplotlib as mat
import matplotlib.pyplot as plt
import scipy.spatial as sp
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from lattice import lattice_3d
from matplotlib.lines import Line2D


#Defining constants(a,b - layer lattice constants , N - size , v - potential constant, theta - layer twist, symmetry - symmetry rotation)
a = 1
b = 1
N = int(33)
v = 1
symmetry = (2/3)*np.pi
m = 10
n = 8
l =(m**2+n**2+4*m*n)/(2*m**2+2*n**2+2*n*m)
theta = np.arccos(l)


#Generate grids:
x = np.linspace(-2,2,100)
y = np.linspace(-2,2,100)
grid = np.meshgrid(x, y)
x_r = np.linspace(-3, 3, 100)
y_r = np.linspace(-3, 3, 100)
grid_r = np.meshgrid(x_r, y_r)


#Cubic:
#basis = np.array([[a,0,0],[0,a,0]])
#basis_top = np.array([[b,0,0],[0,b,0]])


#Hexagonal:
basis = np.array([[a,0,0],[0.5*a,0.5*np.sqrt(3)*a,0]])
basis_top = np.array([[b,0,0],[0.5*b,0.5*np.sqrt(3)*b,0]])
lattice = []
twisted_lattice = []
first_moire = []
lattice_c = lattice_3d(basis, basis_top, N, a, b, theta, grid, v, symmetry, m ,n, first_moire)


#Generate basis:
reciprocal_basis = lattice_c.generate_reciprocal_basis(basis)
twisted_basis = lattice_c.generate_twisted_basis(basis_top, theta)
twisted_reciprocal_basis = lattice_c.generate_reciprocal_basis(twisted_basis)
moire_basis = lattice_c.generate_moire_basis(reciprocal_basis, twisted_reciprocal_basis)
reciprocal_moire_basis = lattice_c.generate_reciprocal_basis(moire_basis)


#Transform from 3D to 2D:
basis = np.delete(basis, 2, 1)
reciprocal_basis = np.delete(reciprocal_basis, 2, 1)
basis_top = np.delete(basis_top, 2, 1)
twisted_basis = np.delete(twisted_basis, 2, 1)
twisted_reciprocal_basis = np.delete(twisted_reciprocal_basis, 2, 1)
moire_basis = np.delete(moire_basis, 2, 1)
reciprocal_moire_basis = np.delete(reciprocal_moire_basis, 2, 1)


#Generate lattice:
lattice = lattice_c.generate_lattice(basis, N)
twisted_lattice = lattice_c.generate_lattice(twisted_basis, N)
reciprocal_lattice = lattice_c.generate_lattice(reciprocal_basis, N)
twisted_reciprocal_lattice = lattice_c.generate_lattice(twisted_reciprocal_basis, N)
moire_lattice = lattice_c.generate_lattice(moire_basis, N)
reciprocal_moire_lattice = lattice_c.generate_lattice(reciprocal_moire_basis, N)


#Generate first reciprocal Moire vectors:
first_moire = lattice_c.generate_first_reciprocal_moire(reciprocal_basis, twisted_reciprocal_basis)


#Generate potential:
V = lattice_c.generate_potential(v, first_moire, theta, grid, symmetry)
#V_reciprocal = lattice_c.generate_reciprocal_potential(v, first_moire, grid, symmetry)


V_const = np.array([-v*np.exp(1j*symmetry), v*np.exp(1j*symmetry), v*np.exp(1j*symmetry), -v*np.exp(1j*symmetry), -v*np.exp(1j*symmetry), v*np.exp(1j*symmetry)])
print(np.sum(np.real(np.exp(1j*(np.tensordot(first_moire[:,1], grid[1])+np.tensordot(first_moire[:,0], grid[0], axes=0)))*V_const[:, None, None], axis=0)))


#Generate Monkhorst geometry:
monkhorst = lattice_c.generate_monkhorst(reciprocal_basis, N)
twisted_monkhorst = lattice_c.generate_monkhorst(twisted_reciprocal_basis, N)


#Generate Voronoi cells:
voronoi = sp.Voronoi(reciprocal_lattice)
voronoi_real = sp.Voronoi(lattice)
twisted_voronoi = sp.Voronoi(twisted_reciprocal_lattice)
twisted_voronoi_real = sp.Voronoi(twisted_lattice)
moire_voronoi = sp.Voronoi(reciprocal_moire_lattice)
moire_voronoi_real = sp.Voronoi(moire_lattice)


#Geometry Plot:
#fig, ax = plt.subplots(figsize=(10,10))
#custom_lines = [Line2D([0], [0], color='r', lw=4),
#                Line2D([0], [0], color='b', lw=4),
#                Line2D([0], [0], color='k', lw=4),
#                Line2D([0], [0], color='purple', lw=4)]
#sp.voronoi_plot_2d(twisted_voronoi, show_vertices=False, show_points=False, line_colors='r', point_size=2, ax=ax)
#sp.voronoi_plot_2d(voronoi, show_vertices=False, show_points=False, line_colors='b', point_size=2, ax=ax)
#sp.voronoi_plot_2d(moire_voronoi, show_vertices=False, show_points=False, line_colors='k', point_size=2, ax=ax)
#plt.quiver(*origin, first_moire[:,0], first_moire[:,1], color = 'purple')
#plt.axis('scaled')
#plt.xlim(-8,8)
#plt.ylim(-8,8)
#ax.legend(custom_lines, ['twisted Voronoi lattice', 'Voronoi lattice', 'Moire Voronoi lattice', 'first reciprocal Moire vectors'])
#plt.title('Hexagonal layers twisted by 7.34°')
#plt.show()

#Moire cell Plot:
#fig, ax = plt.subplots(figsize=(10,10))
#vmin,vmax = 0.0,1.0
#norm = mat.colors.Normalize(vmin=vmin, vmax=vmax)
#custom_lines = [Line2D([0], [0], color='k', lw=4),
#                Line2D([0], [0], color='dimgray', lw=4)]
#sp.voronoi_plot_2d(twisted_voronoi_real, show_vertices=False, show_points=False, line_colors='k', line_width=2, point_size=4, ax=ax)
#sp.voronoi_plot_2d(voronoi_real, show_vertices=False, show_points=False, line_colors='dimgray', point_size=4, line_width=1.5, ax=ax)
#plt.plot(moire_basis[:,0], moire_basis[:,1], "ko", color = 'r', ms=6)
#plt.pcolormesh(grid[0]/a_m, grid[1]/a_m, V, alpha =0.8, norm=norm)
#plt.colorbar()
#plt.axis('scaled')
#plt.xlim(-2,2)
#plt.ylim(-2,2)
#ax.legend(custom_lines, ['twisted Voronoi lattice (real space)', 'Voronoi lattice (real space)'])
#plt.title('Hexagonal layers twisted by 7.34°')
#plt.show()





def calc_moire_potential_reciprocal_on_grid(real_space_points, reciprocal_space_grid, moire_potential_pointwise):
    r"""
    Calculate the reciprocal moire potential on a grid using
    .. math::
        V^{\text{M}}_{G_{\text{M}}} = \frac{1}{A}\int_{\text{MWSC}}
        V_{\text{M}}(\vec{r}\,)\text{e}^{-\text{i}G_{\text{M}}\vec{R}}\text{d}r^2
    with MWSC being the first Moire Wigner Seitz cell.
    :param real_space_points: Real space sample points in the MWSC (for example a Monkhorst-Pack grid)
    :param reciprocal_space_grid: Grid of reciprocal vectors :math:`G_{\text{M}}`
    :param moire_potential_pointwise: Pre-calculated real space Moire potential :math:`V^{\text{M}}(\vec{r}\,)`
    :type real_space_points: numpy.ndarray
    :type reciprocal_space_grid: numpy.ndarray
    :type moire_potential_pointwise: numpy.ndarray
    :rtype: numpy.ndarray
    """

    integrand = np.exp(
            -1j*(
                np.tensordot(real_space_points[:,0], reciprocal_space_grid[0], axes=0) + 
                np.tensordot(real_space_points[:,1], reciprocal_space_grid[1], axes=0)
            ))*moire_potential_pointwise[..., None, None]
    integral = integrand.sum(axis=0)
    return integral/len(real_space_points)