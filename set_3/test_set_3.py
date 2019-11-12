import os
import imp
import sys
import timeout_decorator
import unittest
import math
import numpy as np
from gradescope_utils.autograder_utils.decorators import weight
from gradescope_utils.autograder_utils.json_test_runner import JSONTestRunner
from pydrake.all import MathematicalProgram, Solve, Variables, Polynomial

from pydrake.all import Variable
import pydrake.symbolic as dsym

class TestSetThree_ProblemTwo(unittest.TestCase):
    def setUp(self):
        pass

    @weight(4)
    @timeout_decorator.timeout(1.0)
    def test_problem_2_1(self):
        """Problem 2_1: Finding Q"""
        from set_3_for_testing import problem_2_1_get_Q
        Q = problem_2_1_get_Q()

        eigs = np.linalg.eig(Q)
        for eig in eigs[0]:
            self.assertGreaterEqual(eig, 0., "Q wasn't positive semi-definite!")

        # Sample at many points
        np.random.seed(0)
        def p(x1, x2):
            return 2*x1**4 + 2*x1**3*x2 - x1**2 * x2**2 + 5*x2**4
        def build_basis(x1, x2):
            return np.array([x1**2, x2**2, x1*x2])

        for i in range(100):
            x = np.random.random(2)
            basis = build_basis(x[0], x[1])
            eval_with_Q = np.dot(basis, np.dot(Q, basis))
            eval_with_p = p(x[0], x[1])
            self.assertAlmostEqual(eval_with_Q, eval_with_p,
                msg = "The quadratic form using your Q doesn't "
                "match with p at x_1 = %f, x_2 = %f" % (x[0], x[1]))

class TestSetThree_ProblemThree(unittest.TestCase):
    def setUp(self):
        from inertial_wheel_pendulum import InertialWheelPendulum
        m1 = 1.
        l1 = 1.
        m2 = 2.
        l2 = 2.
        r = 1.0
        g = 10
        input_max = 10
        self.pendulum_plant = InertialWheelPendulum(
            m1 = m1, l1 = l1, m2 = m2, l2 = l2, 
            r = r, g = g, input_max = input_max)


    @weight(2)
    @timeout_decorator.timeout(1.0)
    def test_problem_3_1_A(self):
        """Problem 3_1: Linearization, A matrix"""
        from inertial_wheel_pendulum import InertialWheelPendulum

        uf = np.array([0.])
        xf = np.array([math.pi, 0, 0, 0])
        A, B = self.pendulum_plant.GetLinearizedDynamics(uf, xf)

        self.assertEqual(A.shape[0], 4, "The shape of A is wrong.")
        self.assertEqual(A.shape[1], 4, "The shape of A is wrong.")

        # Take numerical Jacobian to get A and B
        f0 = self.pendulum_plant.evaluate_f(uf, xf)
        A_num = np.zeros((4, 4))
        for axis in range(4):
            xdiff = np.zeros(4)
            xdiff[axis] = 1E-4
            fd = self.pendulum_plant.evaluate_f(uf, xf+xdiff)
            A_num[:, axis] = (fd - f0)/1E-4

        self.assertLessEqual(np.sum(np.abs(A-A_num)), 1E-4, "Your A matrix didn't match a numerically-derived version.")

    @weight(2)
    @timeout_decorator.timeout(1.0)
    def test_problem_3_1_B(self):
        """Problem 3_1: Linearization, B matrix"""
        from inertial_wheel_pendulum import InertialWheelPendulum

        uf = np.array([0.])
        xf = np.array([math.pi, 0, 0, 0])
        A, B = self.pendulum_plant.GetLinearizedDynamics(uf, xf)

        self.assertEqual(B.shape[0], 4, "The shape of B is wrong.")
        self.assertEqual(B.shape[1], 1, "The shape of B is wrong.")

        # Take numerical Jacobian to get A and B
        f0 = self.pendulum_plant.evaluate_f(uf, xf)
        B_num = np.zeros((4, 1))
        for axis in range(1):
            udiff = np.zeros(1)
            udiff[axis] = 1E-4
            fd = self.pendulum_plant.evaluate_f(uf+udiff, xf)
            B_num[:, axis] = (fd - f0)/1E-4

        self.assertLessEqual(np.sum(np.abs(B-B_num)), 1E-4, "Your B matrix didn't match a numerically-derived version.")   


    @weight(2)
    @timeout_decorator.timeout(1.0)
    def test_problem_3_2(self):
        """Problem 3.2: Controllability"""

        from set_3_for_testing import is_controllable

        # Test controllability on a few reasonable test cases
        true_test_cases = [
            [ np.eye(2), np.eye(2) ],
            [ np.zeros((2, 2)), np.eye(2) ],
            self.pendulum_plant.GetLinearizedDynamics([0], [math.pi, 0., 0., 0.]),
            self.pendulum_plant.GetLinearizedDynamics([0], [0., 0., 0., 0.])
        ]

        false_test_cases = [
            [ np.eye(2), np.zeros((2, 1)) ],
            [ np.eye(2), np.ones((2, 1)) ],
            [ np.eye(5), np.zeros((5, 2)) ],
        ]

        for A, B in true_test_cases:
            self.assertTrue(is_controllable(A, B),
                "".join(["Controllability for A = ",
                         np.array_str(A),
                         ", B = ", 
                         np.array_str(B),
                         " should have been True"]))

        for A, B in false_test_cases:
            self.assertFalse(is_controllable(A, B),
                "".join(["Controllability for A = ",
                         np.array_str(A),
                         ", B = ", 
                         np.array_str(B),
                         " should have been False"]))

    def checkConvergenceOfStateLog(self, state_log, should_have_converged=True):
        num_steps = state_log.data().shape[1]

        # Checks that the final state is within a pretty permissive epsilon
        # of the upright fixed point

        def error(x):
            # Penalize deviation in inertial wheel speed much less
            theta_wrapped = (x[0]) % (2 * math.pi )
            return abs(theta_wrapped-math.pi) + abs(x[2]-0.) + 0.01*abs(x[3]-0.)

        final_state_epsilon = 0.1 # Pretty permissive, but will catch divergence
        initial_state = state_log.data()[:, 0]
        final_state = state_log.data()[:, -1]
        final_state_error = error(final_state)
        if should_have_converged:
            self.assertLess(final_state_error, final_state_epsilon,
                "".join([
                    "x0 = ",
                    np.array_str(initial_state),
                    " did not converge to the upright fixed point. ",
                    "Final state was instead ",
                    np.array_str(final_state)
                    ]))
        else:
            self.assertGreater(final_state_error, final_state_epsilon,
                "".join([
                    "x0 = ",
                    np.array_str(initial_state),
                    " converged to the upright fixed point, but we ",
                    "expected it not to. ",
                    "Final state was ",
                    np.array_str(final_state)
                    ]))


    @weight(4)
    @timeout_decorator.timeout(30.0)
    def test_problem_3_3(self):
        """Problem 3.3: LQR"""

        from set_3_for_testing import create_reduced_lqr, lqr_controller
        from inertial_wheel_pendulum import RunSimulation

        A, B = self.pendulum_plant.GetLinearizedDynamics(np.array([0.]), 
            np.array([math.pi, 0., 0., 0.]))

        K, S = create_reduced_lqr(A, B)

        self.assertFalse(np.any(np.isnan(K)), "No elements of K should be NaN")
        self.assertFalse(np.any(np.isnan(S)), "No elements of S should be NaN")

        conditions_that_should_converge = [
            np.array([math.pi, 1000.0, 0.0, 0.0]),
            np.array([math.pi+0.05, 0.0, 0.0, 0.0]),
            np.array([math.pi-0.05, 0.0, 0.1, 0.0]),
            np.array([math.pi, 0.0, 0.0, 1.0]),
        ]

        for x0 in conditions_that_should_converge:
            # Run a forward sim from here
            duration = 1.
            input_log, state_log = RunSimulation(self.pendulum_plant,
                                    lqr_controller,
                                    x0 = x0,
                                    duration = duration)
            self.checkConvergenceOfStateLog(state_log, True)

    @weight(2)
    @timeout_decorator.timeout(30.0)
    def test_problem_3_4(self):
        """Problem 3.4: LQR ROA, Prologue"""

        from set_3_for_testing import lqr_controller
        from set_3_for_testing import get_x0_does_not_converge
        from set_3_for_testing import get_x0_does_converge
        from inertial_wheel_pendulum import RunSimulation

        # Run a forward sim from here
        duration = 2.
        eps = 1E-2 # pretty permissive, but should catch divergences
        x0 = get_x0_does_converge()
        input_log, state_log = RunSimulation(self.pendulum_plant,
                                lqr_controller,
                                x0 = x0,
                                duration = duration)
        self.checkConvergenceOfStateLog(state_log, True)


        x0 = get_x0_does_not_converge()
        input_log, state_log = RunSimulation(self.pendulum_plant,
                                lqr_controller,
                                x0 = x0,
                                duration = duration)
        self.checkConvergenceOfStateLog(state_log, False)

    @weight(2)
    @timeout_decorator.timeout(30.0)
    def test_problem_3_5(self):
        """Problem 3.5: LQR ROA, Evaluating F, V, and Vdot"""
        from set_3_for_testing import calcF, calcV, calcVdot

        # Sample at a couple of points and sanity-check some
        # basic things.
        # Not super-involved testing -- visual checking of the
        # plots is easier and more informative.
        np.random.seed(0)
        for i in range(100):
            sample_x = np.random.random(4)*100 - 50.
            # Make sure they don't return a nan for any of these conditions
            self.assertFalse(np.any(np.isnan(calcF(sample_x))),
                "".join([
                    "calcF(",
                    np.array_str(sample_x),
                    ") returned a NaN"
                    ]))
            if np.sum(np.abs(sample_x)) > 0.0:
                self.assertGreater(calcV(sample_x), 0.0,
                    "".join([
                        "V(",
                        np.array_str(sample_x),
                        ") was nonpositive."
                        ]))
        
        vAtFP = calcV(np.array([math.pi, 0., 0., 0.]))
        self.assertAlmostEqual(vAtFP, 0.0,
            msg="V(pi,0,0,0) = %f != 0.0" % vAtFP)
        vdotAtFP = calcVdot(np.array([math.pi, 0., 0., 0.]))
        self.assertAlmostEqual(vdotAtFP, 0.0,
            msg="Vdot(pi,0,0,0) = %f != 0.0" % vdotAtFP)

    @weight(3)
    @timeout_decorator.timeout(30.0)
    def test_problem_3_6(self):
        """Problem 3.6: LQR ROA, Numerical estimate of the ROA"""

        from inertial_wheel_pendulum import RunSimulation
        # Assume from the run-through that vsamples, vdot samples are good...
        from set_3_for_testing import (
            estimate_rho, V_samples, Vdot_samples, calcV, calcVdot, lqr_controller)

        # But regenerate rho to make sure that wasn't clobbered... more likely
        # to have been, as it's a simpler name and just a scalar
        rho = estimate_rho(V_samples, Vdot_samples)

        self.assertGreater(rho, 0.0, "rho should be bigger than 0.")
        self.assertLess(rho, calcV(np.zeros(4)), "rho can't include (0, 0, 0, 0) due to the input limits.")

        np.random.seed(0)

        # Sample at a few points in the ROA
        # and sanity check that:
        #   Vdot is negative
        #   the LQR simulation converges from this position
        for i in range(10):

            # Rejection sample to find a rho
            # (this might be a bad idea for small rho...)
            sample_v = rho + 1.
            while sample_v >= rho:
                sample_x = np.random.random(4)*10
                sample_v = calcV(sample_x)

            sample_vdot = calcVdot(sample_x)
            self.assertLess(sample_vdot, 0.0,
                "".join([
                    "Vdot sampled at x0=",
                    np.array_str(sample_x),
                    " was positive (Vdot = ",
                    str(sample_vdot),
                    ")"]))

            # Run a forward sim from here
            duration = 10.
            eps = 1E-2 # pretty permissive, but should catch divergences
            input_log, state_log = RunSimulation(self.pendulum_plant,
                                    lqr_controller,
                                    x0 = sample_x,
                                    duration = duration)
            self.checkConvergenceOfStateLog(state_log, True)

    @weight(4)
    @timeout_decorator.timeout(30.0)
    def test_problem_3_8(self):
        """Problem 3.8: Combined Swing-Up and Stabilization"""

        # Assume from the run-through that vsamples, vdot samples are good...
        from inertial_wheel_pendulum import RunSimulation
        from set_3_for_testing import combined_controller

        np.random.seed(0)
        conditions_that_should_converge = [
            np.array([0., 0., 0., 0.]),
            np.array([math.pi, 0., 0., 0.]),
            np.array([3*math.pi, 0., 0., 0.]),
            np.array([0., -100., 0., 0.]),
            np.array([0., 0., 0., 20.]),
            np.random.random(4), # Three random initial conditions for fun
            np.random.random(4),
            np.random.random(4)
        ]

        for x0 in conditions_that_should_converge:
            # Run a forward sim from here
            duration = 30.
            input_log, state_log = RunSimulation(self.pendulum_plant,
                                    combined_controller,
                                    x0 = x0,
                                    duration = duration)
            self.checkConvergenceOfStateLog(state_log, True)


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

def run_tests(local = True,
              notebook_path = "./",
              test_output_file = "test_results.json"):
    try:
        # Check for existence of the expected files
        expected_files = [
            "set_3.ipynb",
            "inertial_wheel_pendulum.py",
            "inertial_wheel_pendulum_visualizer.py"
        ]
        print notebook_path
        for file in expected_files:
            if not os.path.isfile(os.path.join(notebook_path, file)):
                raise ValueError("Couldn't find an expected file: %s" % file)

        os.system("rm -f %s" % test_output_file)
        os.system("rm -f /tmp/set_3.py /tmp/set_3.pyc")
        os.system("jupyter nbconvert --ExecutePreprocessor.timeout=60 --output-dir /tmp --output set_3 --to python %s" % notebook_path + "set_3.ipynb")

        with open("/tmp/set_3.py") as f:
            content = f.readlines()
        filtered_content = []

        string_exclude_list = [
            "%matplotlib",
            "plt.show()",
            "test_set_3",
            "get_ipython"
        ]
        for i in content:
            if not any([s in i for s in string_exclude_list]):
                filtered_content.append(i)
        os.system("rm -f /tmp/set_3_for_testing.py /tmp/set_3_for_testing.pyc")
        os.system("touch /tmp/set_3_for_testing.py")
        with open('/tmp/set_3_for_testing.py', 'a') as the_file:
            for j in filtered_content:
                the_file.write(j+'\n')

        # Import the user support files
        inertial_wheel_pendulum = timeout_decorator.timeout(60)(
            lambda: imp.load_source('inertial_wheel_pendulum', os.path.join(notebook_path, 'inertial_wheel_pendulum.py')))()
        inertial_wheel_pendulum_visualizer = timeout_decorator.timeout(60)(
            lambda: imp.load_source('inertial_wheel_pendulum_visualizer', os.path.join(notebook_path, 'inertial_wheel_pendulum_visualizer.py')))()
        # And import the notebook itself, so we can grab everything it defines.
        # (This runs the entire notebook, as it has been compiled to be a giant script.)
        set_3_for_testing = timeout_decorator.timeout(60)(
            lambda: imp.load_source('set_3_for_testing', '/tmp/set_3_for_testing.py'))()

        do_testing = True

    except timeout_decorator.timeout_decorator.TimeoutError:
        global_fail_with_error_message("Timed out importing your files.", test_output_file)
        do_testing = False
    except Exception as e:
        import traceback
        global_fail_with_error_message("Unknown exception while setting up: " + traceback.format_exc(1), test_output_file)
        do_testing = False

    if do_testing:
        test_cases = [TestSetThree_ProblemTwo, 
                      TestSetThree_ProblemThree]

        suite = unittest.TestSuite()
        for test_class in test_cases:
            tests = unittest.defaultTestLoader.loadTestsFromTestCase(test_class)
            suite.addTests(tests)

        with open(test_output_file, "w") as f:
            JSONTestRunner(stream=f).run(suite)

    os.system("rm -f /tmp/set_3_for_testing.py /tmp/set_3_for_testing.pyc")
    os.system("rm -f /tmp/set_3.py /tmp/set_3.pyc")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Please invoke with two arguments: the ipynb, and the results json to write."
        exit(1)

    run_tests(local=True, notebook_path=sys.argv[1], test_output_file=sys.argv[2])
