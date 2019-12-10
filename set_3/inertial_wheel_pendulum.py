import math
import numpy as np

from pydrake.all import (
    DiagramBuilder,
    Simulator,
    SignalLogger,
    VectorSystem
    )

# Define a system to calculate the continuous dynamics
# of the inertial wheel pendulum.
# 
# This class takes as input the physical description
# of the system, in terms of the center of mass of 
# the first link (m1 centered at l1) and the second
# link (m2 centered at l2, with radius r).
class InertialWheelPendulum(VectorSystem):
    def __init__(self, m1 = 1., l1 = 1., 
                       m2 = 2., l2 = 2., r = 1.0,
                       g = 10., input_max = 10.):
        VectorSystem.__init__(self,
            1,                           # One input (torque at reaction wheel).
            4)                           # Four outputs (theta, phi, dtheta, dphi)
        self._DeclareContinuousState(4)  # Four states (theta, phi, dtheta, dphi).

        self.m1 = float(m1)
        self.l1 = float(l1)
        self.m2 = float(m2)
        self.l2 = float(l2)
        self.r = float(r)
        self.g = float(g)
        self.input_max = float(input_max)

        # Go ahead and calculate rotational inertias.
        # Treat the first link as a point mass.
        self.I1 = self.m1 * self.l1 ** 2
        # Treat the second link as a disk.
        self.I2 = 0.5 * self.m2 * self.r**2

    # This method returns (M, C, tauG, B)
    # according to the dynamics of this system.
    def GetManipulatorDynamics(self, q, qd):
        M = np.array(
            [[self.m1*self.l1**2 + self.m2*self.l2**2 
             + self.I1 + self.I2, self.I2], [self.I2, self.I2]])
        C = np.array([[0, 0], [0, 0]])
        tauG = np.array(
            [[-(self.m1*self.l1 + self.m2*self.l2)*self.g*math.sin(q[0])],
             [0]])
        B = np.array([[0.],
                      [1.]])
        return (M, C, tauG, B)

    # This helper uses the manipulator dynamics to evaluate
    # \dot{x} = f(x, u). It's just a thin wrapper around
    # the manipulator dynamics. If throw_when_limits_exceeded
    # is true, this function will throw a ValueError when
    # the input limits are violated. Otherwise, it'll clamp
    # u to the input range.
    def evaluate_f(self, u, x, throw_when_limits_exceeded=True):
        # Bound inputs
        if throw_when_limits_exceeded and abs(u[0]) > self.input_max:
            raise ValueError("You commanded an out-of-range input of u=%f"
                              % (u[0]))
        else:
            u[0] = max(-self.input_max, min(self.input_max, u[0]))

        # Use the manipulator equation to get qdd.
        q = x[0:2]
        qd = x[2:4]
        (M, C, tauG, B) = self.GetManipulatorDynamics(q, qd)

        # Awkward slice required on tauG to get shapes to agree --
        # numpy likes to collapse the other dot products in this expression
        qdd = np.dot(np.linalg.inv(M), (tauG[:, 0] + np.dot(B, u) - np.dot(C, qd)))

        return np.hstack([qd, qdd])


    # This method calculates the time derivative of the state,
    # which allows the system to be simulated forward in time.
    def _DoCalcVectorTimeDerivatives(self, context, u, x, xdot):
        q = x[0:2]
        qd = x[2:4]
        xdot[:] = self.evaluate_f(u, x, throw_when_limits_exceeded=True)

    # This method calculates the output of the system
    # (i.e. those things that are visible downstream of
    # this system) from the state. In this case, it
    # copies out the full state.
    def _DoCalcVectorOutput(self, context, u, x, y):
        y[:] = x

    # The Drake simulation backend is very careful to avoid
    # algebraic loops when systems are connected in feedback.
    # This system does not feed its inputs directly to its
    # outputs (the output is only a function of the state),
    # so we can safely tell the simulator that we don't have
    # any direct feedthrough.
    def _DoHasDirectFeedthrough(self, input_port, output_port):
        if input_port == 0 and output_port == 0:
            return False
        else:
            # For other combinations of i/o, we will return
            # "None", i.e. "I don't know."
            return None

    # The method return matrices (A) and (B) that encode the
    # linearized dynamics of this system around the fixed point
    # u_f, x_f.
    def GetLinearizedDynamics(self, u_f, x_f):
        q_f = x_f[0:2]
        qd_f = x_f[2:4]

        # You might want at least one of these.
        (M, C_f, tauG_f, B_f) = self.GetManipulatorDynamics(q_f, qd_f)


        '''
        Autograded answer for 3.1: Fill in the rest of this function,
        computing the linearized dynamics of this system around the
        specified point.
        '''
        th1_f = x_f[0]
        tau_jacobian = np.array([[-(self.m1*self.l1+self.m2*self.l2)*self.g*np.cos(th1_f),0],[0,0]])
        H = np.dot(np.linalg.inv(M),tau_jacobian)
        row1 = np.concatenate([np.zeros([2,2]),np.eye(2)],axis=1)
        row2 = np.concatenate([H,np.zeros([2,2])],axis=1)
        A = np.concatenate([row1,row2], axis=0)
        B = np.concatenate([np.zeros([2,1]),np.dot(np.linalg.inv(M),B_f)],axis=0)
        return (A, B)

class PendulumController(VectorSystem):
    ''' System to control the pendulum. Must be handed
    a function with signature:
        u = f(t, x)
    that computes control inputs for the pendulum. '''

    def __init__(self, feedback_rule):
        VectorSystem.__init__(self,
            4,                           # Four inputs: full state inertial wheel pendulum..
            1)                           # One output (torque for reaction wheel).
        self.feedback_rule = feedback_rule

    # This method calculates the output of the system from the
    # input by applying the supplied feedback rule.
    def _DoCalcVectorOutput(self, context, u, x, y):
        # Remember that the input "u" of the controller is the
        # state of the plant
        y[:] = self.feedback_rule(u)


def RunSimulation(pendulum_plant, control_law, x0=np.random.random((4, 1)), duration=30):
    pendulum_controller = PendulumController(control_law)

    # Create a simple block diagram containing the plant in feedback
    # with the controller.
    builder = DiagramBuilder()
    # The last pendulum plant we made is now owned by a deleted
    # system, so easiest path is for us to make a new one.
    plant = builder.AddSystem(InertialWheelPendulum(
        m1 = pendulum_plant.m1,
        l1 = pendulum_plant.l1, 
        m2 = pendulum_plant.m2, 
        l2 = pendulum_plant.l2, 
        r = pendulum_plant.r, 
        g = pendulum_plant.g, 
        input_max = pendulum_plant.input_max))

    controller = builder.AddSystem(pendulum_controller)
    builder.Connect(plant.get_output_port(0), controller.get_input_port(0))
    builder.Connect(controller.get_output_port(0), plant.get_input_port(0))

    # Create a logger to capture the simulation of our plant
    input_log = builder.AddSystem(SignalLogger(1))
    input_log._DeclarePeriodicPublish(0.033333, 0.0)
    builder.Connect(controller.get_output_port(0), input_log.get_input_port(0))

    state_log = builder.AddSystem(SignalLogger(4))
    state_log._DeclarePeriodicPublish(0.033333, 0.0)
    builder.Connect(plant.get_output_port(0), state_log.get_input_port(0))

    diagram = builder.Build()

    # Set the initial conditions for the simulation.
    context = diagram.CreateDefaultContext()
    state = context.get_mutable_continuous_state_vector()
    state.SetFromVector(x0)

    # Create the simulator.
    simulator = Simulator(diagram, context)
    simulator.Initialize()
    simulator.set_publish_every_time_step(False)
    simulator.get_integrator().set_fixed_step_mode(True)
    simulator.get_integrator().set_maximum_step_size(0.005)

    # Simulate for the requested duration.
    simulator.StepTo(duration)

    return input_log, state_log