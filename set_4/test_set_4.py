import os
import imp
import sys
import timeout_decorator
import unittest
import math
import numpy as np
from gradescope_utils.autograder_utils.decorators import weight
from gradescope_utils.autograder_utils.json_test_runner import JSONTestRunner

from pydrake.all import (
    AbstractValue,
    DiagramBuilder,
    Parser,
    PortDataType,
    MultibodyPlant,
    UniformGravityFieldElement,
    Simulator,
)
import pydrake.symbolic as dsym


# persist these trajectory optimizations by giving them global scope
trajectory_optimization_results = []

class TestSetFour_OrbitalTransfer(unittest.TestCase):
    def setUp(self):
        self.initial_states_for_testing = [
            np.asarray([-2.0, 0.4, -0.1, 3.]),
            np.asarray([-2.8, 0.2, 0.1, 1.]),
            np.asarray([-2.0, 0.5, -0.4, 0.]),
            np.asarray([-3.2, 0.2, 0.1, 1.])
        ] 
        self.minimum_time = 5.0
        self.maximum_time = 15.0

    @weight(1)
    @timeout_decorator.timeout(40.0)
    ## the 00 makes it so that this test runs first
    def test_00_run_trajectory_optimization(self):
        """Run a few trajectory optimizations once, and persist the output"""
        from orbital_transfer import OrbitalTransferRocket
        rocket = OrbitalTransferRocket()

        global trajectory_optimization_results

        for state_initial in self.initial_states_for_testing:
            traj, u_traj, time_array = rocket.compute_trajectory_to_other_world(state_initial, self.minimum_time, self.maximum_time)
            trajectory_optimization_results.append([traj, u_traj, time_array])

        self.assertTrue(True, 
            "This means your trajectory optimizations did not solve within the specified time of 30 seconds")

    @weight(1)
    @timeout_decorator.timeout(2.0)
    def test_parameters(self):
        """Check that the original parameters have not been changed"""
        from orbital_transfer import OrbitalTransferRocket
        rocket = OrbitalTransferRocket()
        
        self.assertTrue(rocket.G  == 9.8,  "# gravitational constant")
        self.assertTrue(rocket.M1 == 0.4,  "# mass of first planet")
        self.assertTrue(rocket.M2 == 0.1,  "# mass of second lanet")
        self.assertTrue(np.allclose(rocket.world_1_position, np.asarray([-2.5,-0.1])),"world_1_position")
        self.assertTrue(np.allclose(rocket.world_2_position, np.asarray([ 2.5, 0.1])),"world_2_position")

    @weight(10)
    @timeout_decorator.timeout(2.0)
    def test_initial_state(self):
        """Check that the initial state is valid"""

        for index, state_initial in enumerate(self.initial_states_for_testing):
            traj = trajectory_optimization_results[index][0]
            self.assertTrue(np.allclose(traj[0,:], state_initial), 
                "The initial state of the 'trajectory' does not match \
                the desired initial state")

    @weight(4)
    @timeout_decorator.timeout(2.0)
    def test_time_array(self):
        """Check that the time_array is valid"""

        for index, result in enumerate(trajectory_optimization_results):
            time_array = result[2]
            previous = time_array[0]
            for j in time_array[1:]:
                self.assertTrue(j > previous, "The time_array must be monotonicly increasing in time")
                previous = j

        self.assertTrue(time_array[-1] >= self.minimum_time, "Time was too short")
        self.assertTrue(time_array[-1] <= self.maximum_time, "Time was too long")

    @weight(4)
    @timeout_decorator.timeout(2.0)
    def test_dynamic_constraints(self):
        """Check that the system approximately obeys the original dynamics"""

        for index, result in enumerate(trajectory_optimization_results):
            traj = result[0]
            u_traj = result[1]
            time_array = result[2]

            for j, u in enumerate(u_traj):
                euler_integration = traj[j] + self.rocket_dynamics_test(traj[j],u)*(time_array[j+1]-time_array[j])
                self.assertTrue(abs(((traj[j+1] - euler_integration)**2).sum()) < .001,
                    "The trajectory, input_trajectory, and time_array have been \
                    computed to not be consistent with out euler integration when testing \
                    x0 = {}, at stage = {}".format(np.array_str(self.initial_states_for_testing[index]),j))

    @weight(10)
    @timeout_decorator.timeout(2.0)
    def test_reached_approximate_orbit(self):
        """Check that the trajectory satisfied the specified 'orbit' constraint"""
        from orbital_transfer import OrbitalTransferRocket
        rocket = OrbitalTransferRocket()
        
        for index, result in enumerate(trajectory_optimization_results):
            traj = result[0]
            
            final_position = traj[-1,0:2]
            world_2_position = rocket.world_2_position

            self.assertTrue( abs(((final_position - world_2_position)**2).sum() - 0.5**2) < .01, 
                "Was not correct distance from second world when testing \
                    x0 = %s" % np.array_str(self.initial_states_for_testing[index]))

            final_velocity = traj[-1,2:4]
            self.assertTrue( (final_velocity**2).sum() < 2.01,
                "Final velocity was too fast when testing \
                    x0 = %s" % np.array_str(self.initial_states_for_testing[index]))

    @weight(4)
    @timeout_decorator.timeout(2.0)
    def test_fuel_consumption(self):
        """Check that the trajectory was reasonably efficient with fuel use"""

        for index, result in enumerate(trajectory_optimization_results):
            u_traj = result[1]
            fuel_consumption = (u_traj**2).sum()
            self.assertTrue(fuel_consumption < 90, 
                "One of the tested trajectories caused the rocket to run out of fuel when testing \
                    x0 = %s" % np.array_str(self.initial_states_for_testing[index]))

            
    def two_norm(self, x):
        slack = .001
        return np.sqrt(((x)**2).sum() + slack)

    def rocket_dynamics_test(self, state, u):
        '''
        Copy of dynamics function, for testing
        '''

        rocket_position = state[0:2]
        derivs = np.zeros_like(state)
        derivs[0:2] = state[2:4]

        G  = 9.8  # gravitational constant
        M1 = 0.4  # mass of first planet
        M2 = 0.1  # mass of second lanet
        world_1_position = np.asarray([-2.5,-0.1])
        world_2_position = np.asarray([2.5,0.1])

        derivs[2]  = G * M1 * (world_1_position[0] - rocket_position[0]) / self.two_norm(world_1_position - rocket_position)**3
        derivs[2] += G * M2 * (world_2_position[0] - rocket_position[0]) / self.two_norm(world_2_position - rocket_position)**3
        derivs[2] += u[0]
        derivs[3]  = G * M1 * (world_1_position[1] - rocket_position[1]) / self.two_norm(world_1_position - rocket_position)**3
        derivs[3] += G * M2 * (world_2_position[1] - rocket_position[1]) / self.two_norm(world_2_position - rocket_position)**3
        derivs[3] += u[1]
        
        return derivs


class TestSetFour_PlanarHopper(unittest.TestCase):
    def setUp(self):
        pass

    @weight(1)
    @timeout_decorator.timeout(1.0)
    def test_thigh_torque_return_type(self):
        """Verify the signature of ChooseThighTorque"""
        from hopper_2d import Hopper2dController
        
        builder = DiagramBuilder()
        plant = builder.AddSystem(MultibodyPlant(0.0005))
        parser = Parser(plant)
        parser.AddModelFromFile("raibert_hopper_2d.sdf")
        plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("ground"))
        plant.Finalize()
        controller = Hopper2dController(plant,
                desired_lateral_velocity = 0.0)

        x0 = np.zeros(10)
        x0[1] = 4.   # in air
        x0[4] = 0.5  # feasible leg length
        x0[5] = 0.1  # initial speed

        torquedes = controller.ChooseThighTorque(x0)
        self.assertIsInstance(torquedes, float, 
            "ChooseThighTorque returned a type other than "\
            "float for X0 = %s, desired_lateral_velocity = %f" %
                (np.array_str(x0), controller.desired_lateral_velocity))

        # Try from another desired velocity
        controller.desired_lateral_velocity = -1.0
        torquedes = controller.ChooseThighTorque(x0)
        self.assertIsInstance(torquedes, float, 
            "ChooseThighTorque returned a type other than "\
            "float for X0 = %s, desired_lateral_velocity = %f" %
                (np.array_str(x0), controller.desired_lateral_velocity))

    @weight(10)
    @timeout_decorator.timeout(60.0)
    def test_continues_hopping(self):
        """Verify that the hopper keeps hopping for 10s"""
        from hopper_2d import Simulate2dHopper

        x0 = np.zeros(10)
        x0[1] = 2.   # in air
        x0[4] = 0.5  # feasible leg length
        x0[5] = 0.1  # initial speed

        T = 10

        hopper, controller, state_log = \
            Simulate2dHopper(x0 = x0,
                             duration = T,
                             desired_lateral_velocity = 0.0)

        # Three seconds used as that's safely longer than the
        # typical bouncing period of this system
        # with the default spring / bouncing controller
        index_of_last_three_seconds = \
            np.argmax(state_log.sample_times() > T-3)

        body_z_history = state_log.data()[1, index_of_last_three_seconds:]
        body_theta_history = state_log.data()[2, index_of_last_three_seconds]

        # Full leg extension is 1.5 off the ground
        theta_max_stance_height = 1.5
        z_indicates_a_bounce = \
            np.any(body_z_history > theta_max_stance_height) and \
            np.any(body_z_history <= theta_max_stance_height)


        self.assertTrue(z_indicates_a_bounce,
            "Bouncing appears to have stopped by the last three seconds "
            "of a %f second simulation from x0 = %s, as indicated by "
            "z being either always above, or always below, z=%f." %
            (T, np.array_str(x0), theta_max_stance_height))


    @weight(5)
    @timeout_decorator.timeout(60.0)
    def test_lateral_velocity(self):
        """Verify that the hopper tracks a desired lateral velocity
           while stabilizing theta and hopping"""
        from hopper_2d import Simulate2dHopper

        x0 = np.zeros(10)
        x0[1] = 2.   # in air
        x0[4] = 0.5  # feasible leg length
        x0[5] = 0.1  # initial speed

        T = 10
        
        desired_lateral_velocity = 0.5

        hopper, controller, state_log = \
            Simulate2dHopper(x0 = x0,
                             duration = T,
                             desired_lateral_velocity = desired_lateral_velocity)

        # Three seconds used as that's safely longer than the
        # typical bouncing period of this system
        # with the default spring / bouncing controller
        index_of_last_three_seconds = \
            np.argmax(state_log.sample_times() > T-3)
        body_z_history = state_log.data()[1, index_of_last_three_seconds:]
        body_xd_history = state_log.data()[5, index_of_last_three_seconds:]

        # Look at theta history across all time -- good tracking means
        # this should *never* deviate too wildly
        body_theta_history = state_log.data()[2, :]

        theta_lim = 0.5
        theta_was_stable = \
            np.all(body_theta_history > -theta_lim) and \
            np.all(body_theta_history < theta_lim)

        # Full leg extension is 1.5 off the ground
        theta_max_stance_height = 1.5
        z_indicates_a_bounce = \
            np.any(body_z_history > theta_max_stance_height) and \
            np.any(body_z_history <= theta_max_stance_height)

        # Really liberal window on desired velocity
        min_desired_velocity = desired_lateral_velocity*0.25
        xd_indicates_velocity_tracking = \
            np.all(body_xd_history > min_desired_velocity)

        self.assertTrue(theta_was_stable,
            "Theta was outside of [-%f, %f] during the "
            "last three seconds of a %f second simulation from "
            "x0 = %s, indicating your hopper didn't stabilize theta " 
            "with desired lateral velocity %f." %
            (theta_lim, theta_lim, T, np.array_str(x0),
             desired_lateral_velocity))

        self.assertTrue(z_indicates_a_bounce,
            "Bouncing appears to have stopped by the last three seconds "
            "of a %f second simulation from x0 = %s, as indicated by "
            "z being either always above, or always below, z=%f, with "
            "desired lateral velocity of %f." %
            (T, np.array_str(x0), theta_max_stance_height,
             desired_lateral_velocity))

        self.assertTrue(min_desired_velocity,
            "Velocity was not always > %f during the last three seconds "
            "of a %f second simulation from x0 = %s with desired lateral "
            "velocity %f." %
            (min_desired_velocity, T, np.array_str(x0),
             desired_lateral_velocity))

    @weight(5)
    @timeout_decorator.timeout(60.0)
    def test_stabilizes_theta(self):
        """Verify that the hopper stabilizes both hopping and theta"""
        from hopper_2d import Simulate2dHopper

        x0 = np.zeros(10)
        x0[1] = 2.   # in air
        x0[2] = -0.1  # start tilted a bit
        x0[4] = 0.5  # feasible leg length
        x0[5] = 0.1  # small lateral velocity
        x0[7] = -0.1  # Base running away

        T = 10

        hopper, controller, state_log = \
            Simulate2dHopper(x0 = x0,
                             duration = T,
                             desired_lateral_velocity = 0.0)

        # Three seconds used as that's safely longer than the
        # typical bouncing period of this system
        # with the default spring / bouncing controller
        index_of_last_three_seconds = \
            np.argmax(state_log.sample_times() > T-3)
        body_z_history = state_log.data()[1, index_of_last_three_seconds:]

        # Look at theta history across all time -- good tracking means
        # this should *never* deviate too wildly
        body_theta_history = state_log.data()[2, :]

        theta_lim = 0.5
        theta_was_stable = \
            np.all(body_theta_history > -theta_lim) and \
            np.all(body_theta_history < theta_lim)

        # Full leg extension is 1.5 off the ground
        theta_max_stance_height = 1.5
        z_indicates_a_bounce = \
            np.any(body_z_history > theta_max_stance_height) and \
            np.any(body_z_history <= theta_max_stance_height)

        self.assertTrue(theta_was_stable,
            "Theta was outside of [-%f, %f] during the "
            "last three seconds of a %f second simulation from "
            "x0 = %s, indicating your hopper didn't stabilize theta." %
            (theta_lim, theta_lim, T, np.array_str(x0)))

        self.assertTrue(z_indicates_a_bounce,
            "Bouncing appears to have stopped by the last three seconds "
            "of a %f second simulation from x0 = %s, as indicated by "
            "z being either always above, or always below, z=%f." %
            (T, np.array_str(x0), theta_max_stance_height))


def pretty_format_json_results(test_output_file):
    import json
    import textwrap

    output_str = ""

    try:
        with open(test_output_file, "r") as f:
            results = json.loads(f.read())
 
        total_score_possible = 0.0

        if "tests" in results.keys():
            for test in results["tests"]:
                output_str += "Test %s: " % (test["name"])
                output_str += "%2.2f/%2.2f.\n" % (test["score"], test["max_score"])
                total_score_possible += test["max_score"]
                if "output" in test.keys():
                    output_str += "  * %s\n" % (
                        textwrap.fill(test["output"], 70,
                                      subsequent_indent = "  * "))
                output_str += "\n"

            output_str += "TOTAL SCORE (automated tests only): %2.2f/%2.2f\n" % (results["score"], total_score_possible)

        else:
            output_str += "TOTAL SCORE (automated tests only): %2.2f\n" % (results["score"])
            if "output" in results.keys():
                output_str += "  * %s\n" % (
                        textwrap.fill(results["output"], 70,
                                      subsequent_indent = "  * "))
                output_str += "\n"
    
    except IOError:
        output_str += "No such file %s" % test_output_file
    except Exception as e:
        output_str += "Other exception while printing results file: ", e

    return output_str

def global_fail_with_error_message(msg, test_output_file):
    import json

    results = {"score": 0.0,
               "output": msg}

    with open(test_output_file, 'w') as f:
        f.write(json.dumps(results,
                           indent=4,
                           sort_keys=True,
                           separators=(',', ': '),
                           ensure_ascii=True))

def run_tests(test_output_file = "test_results.json"):
    try:
        # Check for existence of the expected files
        expected_files = [
            "hopper_2d.py",
            "orbital_transfer.py"
        ]
        for file in expected_files:
            if not os.path.isfile(file):
                raise ValueError("Couldn't find an expected file: %s" % file)

        do_testing = True

    except Exception as e:
        import traceback
        global_fail_with_error_message("Somehow failed trying to import the files needed for testing " + traceback.format_exc(1), test_output_file)
        do_testing = False

    if do_testing:
        test_cases = [TestSetFour_OrbitalTransfer,
                      TestSetFour_PlanarHopper]

        suite = unittest.TestSuite()
        for test_class in test_cases:
            tests = unittest.defaultTestLoader.loadTestsFromTestCase(test_class)
            suite.addTests(tests)

        with open(test_output_file, "w") as f:
            JSONTestRunner(stream=f).run(suite)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print "Please invoke with one argument: the result json to write."
        print "(This test file assumes it's in the same directory as the code to be tested."
        exit(1)

    run_tests(test_output_file=sys.argv[1])
