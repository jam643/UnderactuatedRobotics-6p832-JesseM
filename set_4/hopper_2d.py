# -*- coding: utf8 -*-

import argparse
import math
import os.path
import time

import numpy as np

from pydrake.all import (
    RigidBodyTree,
    AddModelInstancesFromSdfString, FloatingBaseType,
    DiagramBuilder, 
    Simulator, VectorSystem,
    ConstantVectorSource, 
    SignalLogger,
    AbstractValue,
    Parser,
    PortDataType,
    MultibodyPlant,
    UniformGravityFieldElement
)
from IPython.display import HTML
import matplotlib.pyplot as plt
from drake import lcmt_viewer_load_robot
from pydrake.common.eigen_geometry import Quaternion
from pydrake.geometry import DispatchLoadMessage, SceneGraph
from pydrake.lcm import DrakeMockLcm
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.systems.rendering import PoseBundle

from underactuated.planar_multibody_visualizer import PlanarMultibodyVisualizer


class Hopper2dController(VectorSystem):
    def __init__(self, hopper, 
        desired_lateral_velocity = 0.0,
        print_period = 0.0):
        # hopper = a rigid body tree representing the 1D hopper
        # desired_lateral_velocity = How fast should the controller
        #  aim to run sideways?
        # print_period = print to console to indicate sim progress
        #  if nonzero
        VectorSystem.__init__(self,
            10, # 10 inputs: x, z, theta, alpha, l, and their derivatives
            2) # 2 outputs: Torque on thigh, and force on the leg extension
               #  link. (The passive spring for on the leg is calculated as
               #  part of this output.)
        self.hopper = hopper
        self.desired_lateral_velocity = desired_lateral_velocity
        self.print_period = print_period
        self.last_print_time = -print_period
        # Remember what the index of the foot is
        # self.foot_body_index = hopper.GetFrameByName("foot").get_body_index()
        self.foot_body_frame = hopper.GetFrameByName("foot")
        self.world_frame = hopper.world_frame()
        
        # The context for the hopper
        self.plant_context = self.hopper.CreateDefaultContext()

        # Default parameters for the hopper -- should match
        # raibert_hopper_1d.sdf, where applicable.
        # You're welcome to use these, but you probably don't need them.
        self.hopper_leg_length = 1.0
        self.m_b = 1.0
        self.m_f = 0.1
        self.l_max = 0.5

        # This is an arbitrary choice of spring constant for the leg.
        self.K_l = 100
           
    def ChooseSpringRestLength(self, X):
        ''' Given the system state X,
            returns a (scalar) rest length of the leg spring.
            We can command this instantaneously, as
            the actual system being simulated has perfect
            force control of its leg extension. '''
        # Unpack states
        x, z, theta, alpha, l = X[0:5]
        zd = X[6]

        # Run out the forward kinematics of the robot
        # to figure out where the foot is in world frame.
        foot_point = np.array([0.0, 0.0, -self.hopper_leg_length])
        foot_point_in_world = self.hopper.CalcPointsPositions(self.plant_context, 
                              self.foot_body_frame, foot_point, self.world_frame)
        in_contact = foot_point_in_world[2] <= 0.01
        
        # Feel free to play with these values!
        # These should work pretty well for this problem set,
        # though.
        if (in_contact):
            if (zd > 0):
                # On the way back up,
                # "push" harder by increasing the effective
                # spring constant.
                l_rest = 1.05
            else:
                # On the way down,
                # "push" less hard by decreasing the effective
                # spring constant.
                l_rest = 1.0
        else:
            # Keep l_rest large to make sure the leg
            # is pushed back out to full extension quickly.
            l_rest = 1.0 

        # See "Hopping in Legged Systems-Modeling and
        # Simulation for the Two-Dimensional One-Legged Case"
        # Section III for a great description of why
        # this works. (It has to do with a natural balance
        # arising between the energy lost from dissipation
        # during ground contact, and the energy injected by
        # this control.)

        return l_rest

    def ChooseThighTorque(self, X):
        ''' Given the system state X,
            returns a (scalar) leg angle torque to exert. '''
        x, z, theta, alpha, l = X[0:5]
        xd, zd, thetad, alphad, ld = X[5:10]
        
        # Run out the forward kinematics of the robot
        # to figure out where the foot is in world frame.
        foot_point = np.array([0.0, 0.0, -self.hopper_leg_length])
        foot_point_in_world = self.hopper.CalcPointsPositions(self.plant_context, 
                              self.foot_body_frame, foot_point, self.world_frame)
        in_contact = foot_point_in_world[2] <= 0.01
        
        # It's all yours from here.
        # Implement a controller that:
        #  - Controls xd to self.desired_lateral_velocity
        #  - Attempts to keep the body steady (theta = 0)
        return 0.0
    
    def DoCalcVectorOutput(self, context, u, x, y):
        # The naming if inputs is confusing, as this is a separate
        # system with its own state (x) and input (u), but the input
        # here is the state of the hopper.
        # Empty now
        if (self.print_period and
            context.get_time() - self.last_print_time >= self.print_period):
            print "t: ", context.get_time()
            self.last_print_time = context.get_time()
        
        # Update the internal context
        plant = self.hopper
        context = self.plant_context
        x_ref = plant.GetMutablePositionsAndVelocities(context)
        x_ref[:] = u
        
        # OK
        l_rest = self.ChooseSpringRestLength(X = u)

        # Passive spring force
        leg_compression_amount = l_rest - u[4]

        y[:] = [  self.ChooseThighTorque(X = u),
                  self.K_l * leg_compression_amount]


'''
Simulates a 2d hopper from initial conditions x0 (which
should be a 10x1 np array) for duration seconds,
targeting a specified lateral velocity and printing to the
console every print_period seconds (as an indicator of
progress, only if print_period is nonzero).
'''
def Simulate2dHopper(x0, duration,
        desired_lateral_velocity = 0.0,
        print_period = 0.0):
    builder = DiagramBuilder()
    
    plant = builder.AddSystem(MultibodyPlant(0.0005))
    scene_graph = builder.AddSystem(SceneGraph())
    plant.RegisterAsSourceForSceneGraph(scene_graph)
    builder.Connect(plant.get_geometry_poses_output_port(),
                    scene_graph.get_source_pose_port(
                        plant.get_source_id()))
    builder.Connect(scene_graph.get_query_output_port(),
                    plant.get_geometry_query_input_port())
    
    # Build the plant
    parser = Parser(plant)
    parser.AddModelFromFile("raibert_hopper_2d.sdf")
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("ground"))
    plant.AddForceElement(UniformGravityFieldElement())
    plant.Finalize()
    
    # Create a logger to log at 30hz
    state_dim = plant.num_positions() + plant.num_velocities()
    state_log = builder.AddSystem(SignalLogger(state_dim))
    state_log.DeclarePeriodicPublish(0.0333, 0.0) # 30hz logging
    builder.Connect(plant.get_continuous_state_output_port(), state_log.get_input_port(0))
    
    # The controller
    controller = builder.AddSystem(
        Hopper2dController(plant,
            desired_lateral_velocity = desired_lateral_velocity,
            print_period = print_period))
    builder.Connect(plant.get_continuous_state_output_port(), controller.get_input_port(0))
    builder.Connect(controller.get_output_port(0), plant.get_actuation_input_port())
    
    # The diagram
    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.Initialize()
    
    plant_context = diagram.GetMutableSubsystemContext(
        plant, simulator.get_mutable_context())
    plant_context.get_mutable_discrete_state_vector().SetFromVector(x0)

    simulator.StepTo(duration)
    return plant, controller, state_log


def ConstructVisualizer():
    from underactuated import PlanarRigidBodyVisualizer
    tree = RigidBodyTree()
    AddModelInstancesFromSdfString(
        open("raibert_hopper_2d.sdf", 'r').read(),
        FloatingBaseType.kFixed,
        None, tree)
    viz = PlanarRigidBodyVisualizer(tree, xlim=[-5, 5], ylim=[-5, 5])
    viz.fig.set_size_inches(10, 5)
    return viz


if __name__ == '__main__':
    x0 = np.zeros(10)
    x0[1] = 2
    x0[4] = 0.5
    hopper, controller, state_log = Simulate2dHopper(x0 = x0,
                               duration=20,
                               desired_lateral_velocity = 0.5,
                               print_period = 1.0)