import math

from pydrake.all import (
    DiagramBuilder,
    Simulator,
    SignalLogger,
    VectorSystem
    )

# Define a system to calculate the continuous dynamics
# of a damped pendulum with a base that is forced to
# move horizontally, with mass m, length l,
# gravity g, and damping b; and base movement
# C * sin(w * t).
class DampedOscillatingPendulumPlant(VectorSystem):
    def __init__(self, m = 3, l = 1, g = 10, b = 2,
                       C = 2, w = 2):
        VectorSystem.__init__(self,
            1,                           # Two inputs (torque at shoulder).
            3)                           # Three outputs (theta, dtheta, base positions)
        self._DeclareContinuousState(2)  # Two state variables (theta, dtheta).
        self.m = float(m)
        self.l = float(l)
        self.g = float(g)
        self.b = float(b)
        self.C = float(C)
        self.w = float(w)

    # This method calculates the time derivative of the state,
    # which allows the system to be simulated forward in time.
    # In this case, it implements the continuous time dynamics
    # of the damped, oscillating-base pendulum.
    def _DoCalcVectorTimeDerivatives(self, context, u, x, xdot):
        # We shouldn't get into a situation where this is
        # necessary... if so, it'll save us time if we just
        # break the simulation.
        if abs(u) > 1E3:
            raise ValueError("Input torque was excessive and would lead"
            " to a really slow simulation. Lower your gains and make sure"
            " your system is stable!")

        theta = x[0]
        theta_dot = x[1]
        t = context.get_time()
        base_position = self.C * math.sin(self.w * context.get_time())


        torque_from_point_mass = \
              -self.m * self.l * self.g * math.sin(theta)
        torque_from_damping = -self.b * theta_dot
        accel_from_base_acceleration = \
            - 1. / self.l * self.w**2 * base_position * math.cos(theta)

        theta_ddot = accel_from_base_acceleration + \
            (torque_from_damping + torque_from_point_mass + u) / (self.m * self.l**2)

        xdot[0] = theta_dot
        xdot[1] = theta_ddot

    # This method calculates the output of the system
    # (i.e. those things that are visible downstream of
    # this system) from the state. In this case, it
    # copies out the full state.
    def _DoCalcVectorOutput(self, context, u, x, y):
        y[0:2] = x
        y[2] = self.C * math.sin(self.w * context.get_time())

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

class PendulumController(VectorSystem):
    ''' System to control the pendulum. Must be handed
    a function with signature:
        u = f(theta, theta_dot)
    that computes control inputs for the pendulum. '''

    def __init__(self, feedback_rule):
        VectorSystem.__init__(self,
            3,                           # Three inputs (theta, dtheta, base position).
            1)                           # One output (torque at shoulder).
        self.feedback_rule = feedback_rule

    # This method calculates the output of the system from the
    # input by applying the supplied feedback rule.
    def _DoCalcVectorOutput(self, context, u, x, y):
        y[:] = self.feedback_rule(context.get_time(), u[0], u[1])


def RunPendulumSimulation(pendulum_plant, pendulum_controller,
                          x0 = [0.9, 0.0], duration = 10):
    '''
        Accepts a pendulum_plant (which should be a
        DampedOscillatingPendulumPlant) and simulates it for 
        'duration' seconds from initial state `x0`. Returns a 
        logger object which can return simulated timesteps `
        logger.sample_times()` (N-by-1) and simulated states
        `logger.data()` (2-by-N).
    '''

    # Create a simple block diagram containing the plant in feedback
    # with the controller.
    builder = DiagramBuilder()
    plant = builder.AddSystem(pendulum_plant)
    controller = builder.AddSystem(pendulum_controller)
    builder.Connect(plant.get_output_port(0), controller.get_input_port(0))
    builder.Connect(controller.get_output_port(0), plant.get_input_port(0))

    # Create a logger to capture the simulation of our plant
    # (We tell the logger to expect a 3-variable input,
    # and hook it up to the pendulum plant's 3-variable output.)
    logger = builder.AddSystem(SignalLogger(3))
    logger._DeclarePeriodicPublish(0.033333, 0.0)

    builder.Connect(plant.get_output_port(0), logger.get_input_port(0))

    diagram = builder.Build()

    # Create the simulator.
    simulator = Simulator(diagram)
    simulator.Initialize()
    simulator.set_publish_every_time_step(False)

    # Set the initial conditions for the simulation.
    state = simulator.get_mutable_context().get_mutable_state()\
                     .get_mutable_continuous_state().get_mutable_vector()
    state.SetFromVector(x0)

    # Simulate for the requested duration.
    simulator.StepTo(duration)

    # Return the logger, which stores the output of the
    # plant across the time steps of the simulation.
    return logger
