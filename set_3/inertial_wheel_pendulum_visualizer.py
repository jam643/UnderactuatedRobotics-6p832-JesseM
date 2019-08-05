# -*- coding: utf8 -*-

import numpy as np
import math
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from pydrake.all import (
        Context,
        DiagramBuilder,
        LeafSystem,
        PortDataType,
    )
import scipy.interpolate

from underactuated import PyPlotVisualizer
from inertial_wheel_pendulum import *

# Custom visualizer for a pendulum with a moving
# base. Very similar to the Underactuated textbook
# Pendulum visualizer, but expects a second input
# specifying the horizontal offset of the base
# of the pendulum.

def populate_square_vertices(edge_length):
    return np.array([[-edge_length, -edge_length, edge_length, edge_length, -edge_length],
                     [-edge_length, edge_length, edge_length, -edge_length, -edge_length]])

def populate_disk_vertices(radius, width, N):
    av = np.linspace(0, 2*math.pi, N)

    outer_circle_x = np.array([radius*math.cos(v) for v in av])
    outer_circle_y = np.array([radius*math.sin(v) for v in av])
    inner_circle_x = np.array([(radius-width)*math.cos(v) for v in av])
    inner_circle_y = np.array([(radius-width)*math.sin(v) for v in av])

    all_x = np.hstack([outer_circle_x, inner_circle_x[::-1], outer_circle_x[0]])
    all_y = np.hstack([outer_circle_y, inner_circle_y[::-1], outer_circle_y[0]])
    return np.vstack([all_x, all_y])

def rotmat(theta):
    return np.array([[math.cos(theta), -math.sin(theta)],
                     [math.sin(theta), math.cos(theta)]])

class InertialWheelPendulumVisualizer(PyPlotVisualizer):
    
    # Visualize base with a small square at origin
    base_size = 0.2

    # Visualize arm with another rectangle
    arm_width = 0.2

    # Visualize the reaction wheel with an outer circle +
    # one crossbar
    crossbar_width = 0.1
    wheel_width = 0.1


    def __init__(self, inertial_wheel_pendulum):
        PyPlotVisualizer.__init__(self)
        self.set_name('Inertial Wheel Pendulum Visualizer')
        self._DeclareInputPort(PortDataType.kVectorValued, 1) # full output of its controller
        self._DeclareInputPort(PortDataType.kVectorValued, 4) # full output of the pendulum visualizer

        lim = (inertial_wheel_pendulum.l2+inertial_wheel_pendulum.r)*2.0
        self.ax.set_xlim([-lim, lim])
        self.ax.set_ylim([-lim, lim])

        self.base_pts = populate_disk_vertices(self.base_size, 0.01, 10)
        self.base = self.ax.fill(self.base_pts[0, :], self.base_pts[1, :], zorder=1, facecolor=(.3, .6, .4), edgecolor='k', closed=True)

        self.arm_pts = populate_square_vertices(1.)
        self.arm_pts[0, :] *= self.arm_width
        self.arm_pts[1, :] = (self.arm_pts[1, :] - 1.) * inertial_wheel_pendulum.l2 / 2
        self.arm = self.ax.fill(self.arm_pts[0, :], self.arm_pts[1, :], zorder=1, facecolor=(.6, .1, 0), edgecolor='k', closed=True)

        self.arm_com_pt = np.array([0, -inertial_wheel_pendulum.l1])
        self.arm_com = self.ax.plot(self.arm_com_pt[0], self.arm_com_pt[1], zorder=2, color='b', marker='o', markersize=4)

        self.flywheel_crossbar_pts = populate_square_vertices(1.)
        self.flywheel_crossbar_pts[1, :] *= inertial_wheel_pendulum.r - self.wheel_width
        self.flywheel_crossbar_pts[0, :] *= self.crossbar_width
        # need to keep this around to handle stacked tfs...
        # I really need to write a shape-generalized, tree-friendly planar rigid body visualizer
        self.flywheel_crossbar_offset = -inertial_wheel_pendulum.l2
        self.flywheel_crossbar = self.ax.fill(self.flywheel_crossbar_pts[0, :], self.flywheel_crossbar_pts[1, :]+self.flywheel_crossbar_offset, zorder=3, facecolor=(.9, .4, 0), edgecolor='k', closed=True)

        self.flywheel_disc_pts = populate_disk_vertices(inertial_wheel_pendulum.r, self.wheel_width, 20)
        self.flywheel_disc_pts[1, :] -= inertial_wheel_pendulum.l2
        self.flywheel_disc = self.ax.fill(self.flywheel_disc_pts[0, :], self.flywheel_disc_pts[1, :], zorder=3, facecolor=(.9, .4, 0), edgecolor='k', closed=True)

        # todo: input visualization?

    def draw(self, context):
        if isinstance(context, Context):
            theta = self.EvalVectorInput(context, 1).get_value()[0]
            phi = self.EvalVectorInput(context, 1).get_value()[1]
        else:
            theta = context[1][0]
            phi = context[1][1]

        rotmat_theta = rotmat(theta)
        rotmat_phi = rotmat(phi)

        path = self.arm[0].get_path()
        path.vertices[:,:] = np.dot(rotmat_theta, self.arm_pts).T

        com_new = np.dot(rotmat_theta, self.arm_com_pt)
        self.arm_com[0].set_data(com_new[0], com_new[1])

        path = self.flywheel_disc[0].get_path()
        path.vertices[:,:] = np.dot(rotmat_theta, self.flywheel_disc_pts).T

        path = self.flywheel_crossbar[0].get_path()
        rotated_crossbar_pts = np.dot(rotmat_phi, self.flywheel_crossbar_pts)
        rotated_crossbar_pts[1, :] += self.flywheel_crossbar_offset
        path.vertices[:,:] = np.dot(rotmat_theta, rotated_crossbar_pts).T


    def animate(self, input_log, state_log, rate, resample=True, repeat=False):
        # log - a reference to a pydrake.systems.primitives.SignalLogger that
        # constains the plant state after running a simulation.
        # rate - the frequency of frames in the resulting animation
        # resample -- should we do a resampling operation to make
        # the samples more consistent in time? This can be disabled
        # if you know the sampling rate is exactly the rate you supply
        # as an argument.
        # repeat - should the resulting animation repeat?
        t = state_log.sample_times()
        u = input_log.data()
        x = state_log.data()

        if resample:
            t_resample = np.arange(0, t[-1], 1./rate)
            u = scipy.interpolate.interp1d(input_log.sample_times(), u, kind='linear', axis=1)(t_resample)
            x = scipy.interpolate.interp1d(t, x, kind='linear', axis=1)(t_resample)
            t = t_resample

        animate_update = lambda i: self.draw([u[:, i], x[:, i]])
        ani = animation.FuncAnimation(self.fig, animate_update, t.shape[0], interval=1000./rate, repeat=repeat)
        return ani