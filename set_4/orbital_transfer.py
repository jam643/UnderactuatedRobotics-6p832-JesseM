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
        fig, axes = plt.subplots(nrows=1,ncols=1,figsize = (16,9))
        axes.plot(rocket_position_x, rocket_position_y,'.-')
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

        # length of horizon (excluding init state)
        N = 50
        trajectory = np.zeros((N+1,4))
        input_trajectory = np.ones((N,2))*10.0
        
        ### My implementation of Direct Transcription 
        # (states and control are all decision vars using Euler integration)
        mp = MathematicalProgram()
        
        # let trajectory duration be a decision var
        total_time = mp.NewContinuousVariables(1,"total_time")
        dt = total_time[0]/N
        
        # create the control decision var (m*N) and state decision var (n*[N+1])
        idx = 0
        u_list = mp.NewContinuousVariables(2,"u_{}".format(idx))
        state_list = mp.NewContinuousVariables(4,"state_{}".format(idx))
        state_list = np.vstack((state_list, mp.NewContinuousVariables(4,"state_{}".format(idx+1))))
        for idx in range(1, N):
            u_list = np.vstack((u_list, mp.NewContinuousVariables(2,"u_{}".format(idx))))
            state_list = np.vstack((state_list, mp.NewContinuousVariables(4,"state_{}".format(idx+1))))
        
        ### Constraints
        # set init state as constraint on stage 0 decision vars
        for state_idx in range(4):
            mp.AddLinearConstraint(state_list[0,state_idx] == state_initial[state_idx])

        # interstage equality constraint on state vars via Euler integration
        # note: Direct Collocation could improve accuracy for same computation
        for idx in range(1, N+1):
            state_new = state_list[idx-1,:] + dt*self.rocket_dynamics(state_list[idx-1,:],u_list[idx-1,:])
            for state_idx in range(4):
                mp.AddConstraint(state_list[idx,state_idx] == state_new[state_idx])
                
        # constraint on time
        mp.AddLinearConstraint(total_time[0] <= maximum_time)
        mp.AddLinearConstraint(total_time[0] >= minimum_time)
        
        # constraint on final state distance (squared)to second planet
        final_dist_to_world_2_sq = (self.world_2_position[0]-state_list[-1,0])**2 + (self.world_2_position[1]-state_list[-1,1])**2
        mp.AddConstraint(final_dist_to_world_2_sq <= 0.25)
        
        # constraint on final state speed (squared
        final_speed_sq = state_list[-1,2]**2 + state_list[-1,3]**2
        mp.AddConstraint(final_speed_sq <= 1)
        
        ### Cost
        # equal cost on vertical/horizontal accels, weight shouldn't matter since this is the only cost
        mp.AddQuadraticCost(1 * u_list[:,0].dot(u_list[:,0]))
        mp.AddQuadraticCost(1 * u_list[:,1].dot(u_list[:,1]))
        
        ### Solve and parse
        result = Solve(mp)
        trajectory = result.GetSolution(state_list)
        input_trajectory = result.GetSolution(u_list)
        tf = result.GetSolution(total_time)
        time_array = np.linspace(0,tf[0],N+1)
        
        print "optimization successful: ", result.is_success()
        print "total num decision vars (x: (N+1)*4, u: 2N, total_time: 1): {}".format(mp.num_vars())
        print "solver used: ", result.get_solver_id().name()
        print "optimal trajectory time: {:.2f} sec".format(tf[0])
        
        return trajectory, input_trajectory, time_array

        

