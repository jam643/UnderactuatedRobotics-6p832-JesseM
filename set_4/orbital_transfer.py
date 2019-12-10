# This is temporary
def transfer_orbits():
    return 1

from numpy import sin, cos
import numpy as np

from pydrake.all import MathematicalProgram, Solve, IpoptSolver, SolverOptions

# These are only for plotting
import matplotlib.animation as animation
from matplotlib.patches import Circle
import matplotlib.pyplot as plt

class OrbitalTransferRocket():

    def __init__(self):
        self.G  = 9.8  # gravitational constant
        self.M1 = 0.4  # mass of first planet
        self.M2 = 0.1  # mass of second lanet
        self.world_1_position = np.asarray([-2.5,-0.1])
        self.world_1_color    = "black"   # change to 'darkgreen' for photorealistic Earth sim
        self.world_2_position = np.asarray([2.5,0.1])
        self.world_2_color    = "black"   # change to 'orangered' for photorealistic Mars sim

    def rocket_dynamics(self, state, u):
        '''
        Calculates the dynamics, i.e.:
           \dot{state} = f(state,u)

        for the rocket + two planets system.

        :param state: numpy array, length 4, comprising state of system:
            [x, y, \dot{x}, \dot{y}]
        :param u: numpy array, length 2, comprising control input for system:
            [\ddot{x}_u, \ddot{y}_u]   
            Note that this is only the added acceleration, note the total acceleration.

        :return: numpy array, length 4, comprising the time derivative of the system state:
            [\dot{x}, \dot{y}, \ddot{x}, \ddot{y}]
        '''

        '''
        NOTE: For the problem set, this function should be left untouched, or at least returned to original form
        in order to pass tests.
        '''

        rocket_position = state[0:2]
        derivs = np.zeros_like(state)
        derivs[0:2] = state[2:4]

        # these local copies of variables make the equations below
        # easier to look at
        G = self.G; M1 = self.M1; M2 = self.M2;
        world_1_position = self.world_1_position; world_2_position = self.world_2_position;

        derivs[2]  = G * M1 * (world_1_position[0] - rocket_position[0]) / self.two_norm(world_1_position - rocket_position)**3
        derivs[2] += G * M2 * (world_2_position[0] - rocket_position[0]) / self.two_norm(world_2_position - rocket_position)**3
        derivs[2] += u[0]
        derivs[3]  = G * M1 * (world_1_position[1] - rocket_position[1]) / self.two_norm(world_1_position - rocket_position)**3
        derivs[3] += G * M2 * (world_2_position[1] - rocket_position[1]) / self.two_norm(world_2_position - rocket_position)**3
        derivs[3] += u[1]
        
        return derivs

    def passive_rocket_dynamics(self, state):
        '''
        Caculates the dynamics with no control input, see documentation for rocket_dynamics
        '''
        u = np.zeros(2)
        return self.rocket_dynamics(state, u)

    def two_norm(self, x):
        '''
        Euclidean norm but with a small slack variable to make it nonzero.
        This helps the nonlinear solver not end up in a position where
        in the dynamics it is dividing by zero.

        :param x: numpy array of any length (we only need it for length 2)
        :return: numpy.float64
        '''
        slack = .001
        return np.sqrt(((x)**2).sum() + slack)

    def simulate_states_over_time(self, state_initial, time_array, input_trajectory):
        '''
        Given an initial state, simulates the state of the system.

        This uses simple Euler integration.  The purpose here of not
        using fancier integration is to provide what will be useful reference for
        a simple direct transcription trajectory optimization implementation.

        The first time of the time_array __is__ the time of the state_initial.

        :param state_initial: numpy array of length 4, see rocket_dynamics for documentation
        :param time_array: numpy array of length N+1 (0, ..., N) whose elements are samples in time, i.e.:
            [ t_0,
              ...
              t_N ] 
            Note the times do not have to be evenly spaced
        :param input_trajectory: numpy 2d array of N rows (0, ..., N-1), and 2 columns, corresponding to
            the control inputs at each time, except the last time, i.e.:
            [ [u_0, u_1],
              ...
              [u_{N-1}, u_{N-1}] ]

        :return: numpy 2d array where the rows are samples in time corresponding
            to the time_array, and each row is the state at that time, i.e.:
            [ [x_0, y_0, \dot{x}_0, \dot{y}_0],
              ...
              [x_N, y_N, \dot{x}_N, \dot{y}_N] ]
        '''
        states_over_time = np.asarray([state_initial])
        for i in range(1,len(time_array)):
            time_step = time_array[i] - time_array[i-1]
            state_next = states_over_time[-1,:] + time_step*self.rocket_dynamics(states_over_time[-1,:], input_trajectory[i-1,:])
            states_over_time = np.vstack((states_over_time, state_next))
        return states_over_time

    def simulate_states_over_time_passive(self, state_initial, time_array):
        '''
        Given an initial state, simulates the state of the system passively

        '''
        input_trajectory = np.zeros((len(time_array)-1,2))
        return self.simulate_states_over_time(state_initial, time_array, input_trajectory)
        
    def plot_trajectory(self, trajectory):
        '''
        Given a trajectory, plots this trajectory over time.

        :param: trajectory: the output of simulate_states_over_time, or equivalent
            Note: see simulate_states_over_time for documentation of the shape of the output
        '''
        input_trajectory = np.zeros((trajectory.shape[0],2))
        self.plot_trajectory_with_boosters(trajectory, input_trajectory)

    def plot_trajectory_with_boosters(self, trajectory, input_trajectory):
        '''
        Given a trajectory and an input_trajectory, plots this trajectory and control inputs over time.

        :param: trajectory: the output of simulate_states_over_time, or equivalent
            Note: see simulate_states_over_time for documentation of the shape of the output
        :param: input_trajectory: the input to simulate_states_over_time, or equivalent
            Note: see simulate_states_over_time for documentation of the shape of the input_trajectory
        '''
        rocket_position_x = trajectory[:,0]
        rocket_position_y = trajectory[:,1]
        fig, axes = plt.subplots(nrows=1,ncols=1)
        axes.plot(rocket_position_x, rocket_position_y)
        circ = Circle(self.world_1_position, radius=0.2, facecolor=self.world_1_color, edgecolor='black', fill=True ,linewidth = 1.0, linestyle='solid')
        axes.add_patch(circ)
        circ = Circle(self.world_2_position, radius=0.1, facecolor=self.world_2_color, edgecolor='black', fill=True ,linewidth = 1.0, linestyle='solid')
        axes.add_patch(circ)
        axes.axis('equal')

        ## if we have an input trajectory, plot it
        if len(input_trajectory.nonzero()[0]):
            # the quiver plot works best with not too many arrows
            max_desired_arrows = 40
            num_time_steps = input_trajectory.shape[0]

            if num_time_steps < max_desired_arrows:
                downsample_rate = 1 
            else: 
                downsample_rate = num_time_steps / max_desired_arrows

            rocket_position_x = rocket_position_x[:-1] # don't need the last state, no control input for it
            rocket_position_y = rocket_position_y[:-1]
            rocket_booster_x = input_trajectory[::downsample_rate,0]
            rocket_booster_y = input_trajectory[::downsample_rate,1]
            Q = plt.quiver(rocket_position_x[::downsample_rate], rocket_position_y[::downsample_rate], \
                rocket_booster_x, rocket_booster_y, units='width', color="red")

        plt.show()

    def compute_trajectory_to_other_world(self, state_initial, minimum_time, maximum_time):
        '''
        Your mission is to implement this function.

        A successful implementation of this function will compute a dynamically feasible trajectory
        which satisfies these criteria:
            - Efficiently conserve fuel
            - Reach "orbit" of the far right world
            - Approximately obey the dynamic constraints
            - Begin at the state_initial provided
            - Take no more than maximum_time, no less than minimum_time

        The above are defined more precisely in the provided notebook.

        Please note there are two return args.

        :param: state_initial: :param state_initial: numpy array of length 4, see rocket_dynamics for documentation
        :param: minimum_time: float, minimum time allowed for trajectory
        :param: maximum_time: float, maximum time allowed for trajectory

        :return: three return args separated by commas:

            trajectory, input_trajectory, time_array

            trajectory: a 2d array with N rows, and 4 columns. See simulate_states_over_time for more documentation.
            input_trajectory: a 2d array with N-1 row, and 2 columns. See simulate_states_over_time for more documentation.
            time_array: an array with N rows. 

        '''
    
        print "Function not yet implemented"

        N = 50
        trajectory = np.zeros((N+1,4))
        input_trajectory = np.ones((N,2))*10.0
        time_used = 100.0
        time_array = np.arange(0.0, time_used, time_used/(N+1))
        return trajectory, input_trajectory, time_array

        

