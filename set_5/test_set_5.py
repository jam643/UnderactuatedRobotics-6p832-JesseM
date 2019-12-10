import os
import imp
import sys
import timeout_decorator
import unittest
import math
import numpy as np
import random
from gradescope_utils.autograder_utils.decorators import weight
from gradescope_utils.autograder_utils.json_test_runner import JSONTestRunner

from pydrake.all import (RigidBodyTree, AddModelInstancesFromSdfString,
                         FloatingBaseType, Variable)
import pydrake.symbolic as dsym


class TestSetFive_GraspMetrics(unittest.TestCase):
    @weight(5)
    @timeout_decorator.timeout(20.0)
    def test_00_achieves_force_closure(self):
        """Test force closure metric on some predefined grasps"""
        from grasp_metrics import achieves_force_closure

        points  = [np.asarray([-1.0,0.]), np.asarray([1.0,0.])]
        normals = [np.asarray([1.0,0.]), np.asarray([-1.0,0.])]
        mu = 0.2
        FC = achieves_force_closure(points, normals, mu)
        self.assertTrue(FC, 
            "This means that one of simple force closure checks failed ")

        r1 = np.asarray([0.1, 1])
        r2 = np.asarray([0.3,-0.4])
        r3 = np.asarray([-0.7,-0.5])
        points = [r1, r2, r3]
        n1 = np.asarray([-0.1,-1.1])
        n1 = n1 / np.linalg.norm(n1)
        n2 = np.asarray([-0.4,1.1])
        n2 = n2 / np.linalg.norm(n2)
        n3 = np.asarray([0.8,1.1])
        n3 = n3 / np.linalg.norm(n3)
        normals = [n1, n2, n3]
        mu = 1.5
        FC = achieves_force_closure(points, normals, mu)
        self.assertTrue(FC, 
            "This means that one of simple force closure checks failed ")

        mu = 1e-7
        FC = achieves_force_closure(points, normals, mu)
        self.assertFalse(FC,
            "Friction cone constraint is probably not properly enforced. ")

        points  = [np.asarray([-1.0,0.]), np.asarray([1.0,0.])]
        normals = [np.asarray([-1.0,0.]), np.asarray([-1.0,0.])]
        mu = 0.2
        FC = achieves_force_closure(points, normals, mu)
        self.assertTrue(not FC, 
            "This means that one of simple force closure checks failed ")

    @weight(10)
    @timeout_decorator.timeout(20.0)
    def test_01_achieves_force_closure(self):
        """Test force closure metric on some random grasps"""
        random.seed(42)
        np.random.seed(42)

        from grasp_metrics import achieves_force_closure
        for i in range(100):
            random_thetas = [np.random.rand()*np.pi for _ in range(2)]
            random_points = [np.array([np.sin(theta), np.cos(theta)]) for theta in random_thetas]
            normals      = [-x/np.linalg.norm(x) for x in random_points]
            if not np.allclose(normals[0],-normals[1]):
                FC = achieves_force_closure(random_points, normals, 1e-7)
                self.assertFalse(FC,
                    "This means that you are computing force closure for two points not antipodal "
                    "(for two points and tiny friction, they need to be opposite to achieve force closure). ")

        for i in range(3):
            random_thetas = [np.random.rand()*2*np.pi for _ in range(100)]
            random_thetas[-1] = random_thetas[-2] + np.pi # One is directly opposite
            random_points = [np.array([np.sin(theta), np.cos(theta)]) for theta in random_thetas]
            normals      = [-x/np.linalg.norm(x) for x in random_points]
            FC = achieves_force_closure(random_points, normals, 0.1)
            self.assertTrue(FC, 
                    "These many grasp points should achieve force closure since "
                    "One of them was chosen to be directly antipoldal. ")  
            negative_normals = [-normal for normal in normals]

        for i in range(3):
            random_thetas = [np.random.rand()*np.pi/3.0 for _ in range(100)]
            random_points = [np.array([np.sin(theta), np.cos(theta)]) for theta in random_thetas]
            normals      = [-x/np.linalg.norm(x) for x in random_points]
            FC = achieves_force_closure(random_points, normals, 0.001)
            negative_normals = [-normal for normal in normals]        
            FC = achieves_force_closure(random_points, negative_normals, 0.1)
            self.assertFalse(FC,
                    "We can't be pulling on the objects, only pushing. "
                    "Recommend checking your f_{i,z} > 0 constraint.") 

    @weight(5)
    @timeout_decorator.timeout(10.0)
    def test_02_compute_convex_hull_volume_two_points(self):
        """Tests the convex hull volume is 0 for two points"""
        from grasp_metrics import compute_convex_hull_volume

        points = [np.asarray([-1.0,0.]), np.asarray([1.0,0.])]
        volume = compute_convex_hull_volume(points)
        self.assertTrue(abs(volume) < 1e-9, "The volume of two points"
            "should be zero, with no error")

    @weight(5)
    @timeout_decorator.timeout(10.0)
    def test_03_compute_convex_hull_volume(self):
        """Tests the convex hull volume is correctly computed on a small sample set"""
        random.seed(42)
        np.random.seed(42)
        from grasp_metrics import compute_convex_hull_volume

        points = [np.asarray([-1.0,0.]), np.asarray([1.0,0.]), np.asarray([0.0001,0.0001])]
        volume = compute_convex_hull_volume(points)
        self.assertTrue(abs(volume) < 1e-2, "This test should have a tiny volume"
            "it just has points: "+str(points) )

        points = [np.asarray([0,0]), np.asarray([1.0,0]), np.asarray([0,1]), np.asarray([1,1])]
        volume = compute_convex_hull_volume(points)
        self.assertTrue(abs(volume - 1) < 1e-3, "This test should have a volume of 1"
            "it has points: "+str(points))

        for i in range(100):
            points.append(np.random.rand(2))
        volume = compute_convex_hull_volume(points)
        self.assertTrue(abs(volume - 1) < 1e-3, "This test should still have a volume of 1"
            "The points that were added are only on the interior.")


def sigmoid(x):
  return 1 / (1 + math.exp(-x))
def test_manipuland_trajectory_callback_A(t):
    return np.array([1.5, 0.5 * sigmoid(t - 2.5), 0.5 * sigmoid(t - 2.5)])
def test_manipuland_trajectory_callback_B(t):
    return np.array([1.5, -0.5 * sigmoid(t - 2.5), 0.5 * sigmoid(t - 2.5)])
def test_manipuland_trajectory_callback_C(t):
    return np.array([1.5 - 0.5 * sigmoid(t - 2.5), 0.5, 0.0])
def calculate_average_cf_over_last_second(contact_log):
    ts = contact_log.sample_times()
    contact_results = contact_log.data()
    average_cf_over_time = np.zeros(len(ts))
    for i, cr in enumerate(contact_results):
        if len(cr) > 0:
            total_cf = 0.0
            for id_1, id_2, r, f in cr:
                total_cf += np.linalg.norm(f)
            total_cf /= len(cr)
            average_cf_over_time[i] = total_cf
    return average_cf_over_time[np.where(ts > (ts[-1] - 1.0))[0]]

class TestSetFive_ManipulationMpc(unittest.TestCase):
    initial_manipuland_pose = np.array([1.5, 0., 0.])

    @weight(10)
    @timeout_decorator.timeout(30.0)
    def test_01_two_finger_A(self):
        """Test two finger grip of the box under motion A."""
        import planar_hand
        num_fingers = 2
        num_hand_q = num_fingers * 3
        
        hand, plant, controller, state_log, contact_log = \
            planar_hand.SimulateHand(
                num_fingers=num_fingers,
                manipuland_sdf="models/manipuland_box.sdf",
                initial_manipuland_pose=self.initial_manipuland_pose,
                manipuland_trajectory_callback = test_manipuland_trajectory_callback_A,
                duration=10,
                mu=0.5,
                n_grasp_search_iters=100,
                control_period = 0.0333)

        f = state_log.data()[num_hand_q:(num_hand_q+3), -1]
        fg = test_manipuland_trajectory_callback_A(10.)
        max_error = np.max(np.abs(f - fg))
        self.assertTrue(max_error <= 0.25,
                        "The final pose error was greater than 0.25 in "
                        "at least one dimension: goal [%f,%f,%f], target "
                        "[%f,%f,%f]." % (fg[0], fg[1], fg[2], f[0], f[1], f[2]))
        
        final_cf = calculate_average_cf_over_last_second(contact_log)
        min_cf = np.min(final_cf)
        self.assertTrue(min_cf >= 0.1,
                        "The average contact force dipped to %f during "
                        "the last second of simulation, while it should "
                        "have stayed above 0.1." % min_cf)

    @weight(10)
    @timeout_decorator.timeout(30.0)
    def test_03_three_finger_B(self):
        """Test three finger grip of the triangle under motion B."""
        import planar_hand
        num_fingers = 3
        num_hand_q = num_fingers * 3
                
        hand, plant, controller, state_log, contact_log = \
            planar_hand.SimulateHand(
                num_fingers=num_fingers,
                manipuland_sdf="models/manipuland_triangle.sdf",
                initial_manipuland_pose=self.initial_manipuland_pose,
                manipuland_trajectory_callback = test_manipuland_trajectory_callback_B,
                duration=10,
                mu=0.5,
                n_grasp_search_iters=100,
                control_period = 0.0333)
        
        f = state_log.data()[num_hand_q:(num_hand_q+3), -1]
        fg = test_manipuland_trajectory_callback_B(10.)
        max_error = np.max(np.abs(f - fg))
        self.assertTrue(max_error <= 0.25,
                        "The final pose error was greater than 0.25 in "
                        "at least one dimension: goal [%f,%f,%f], target "
                        "[%f,%f,%f]." % (fg[0], fg[1], fg[2], f[0], f[1], f[2]))
                
        final_cf = calculate_average_cf_over_last_second(contact_log)
        min_cf = np.min(final_cf)
        self.assertTrue(min_cf >= 0.1,
                        "The average contact force dipped to %f during "
                        "the last second of simulation, while it should "
                        "have stayed above 0.1." % min_cf)

    @weight(10)
    @timeout_decorator.timeout(30.0)
    def test_02_four_finger_B(self):
        """Test four finger grip of the large ball under motion C."""
        import planar_hand
        num_fingers = 4
        num_hand_q = num_fingers * 3
        
        hand, plant, controller, state_log, contact_log = \
            planar_hand.SimulateHand(
                num_fingers=num_fingers,
                manipuland_sdf="models/manipuland_ball_large.sdf",
                initial_manipuland_pose=self.initial_manipuland_pose,
                manipuland_trajectory_callback = test_manipuland_trajectory_callback_C,
                duration=10,
                mu=0.5,
                n_grasp_search_iters=100,
                control_period = 0.0333)

        f = state_log.data()[num_hand_q:(num_hand_q+3), -1]
        fg = test_manipuland_trajectory_callback_C(10.)
        max_error = np.max(np.abs(f - fg))
        self.assertTrue(max_error <= 0.25,
                        "The final pose error was greater than 0.25 in "
                        "at least one dimension: goal [%f,%f,%f], target "
                        "[%f,%f,%f]." % (fg[0], fg[1], fg[2], f[0], f[1], f[2]))
        
        final_cf = calculate_average_cf_over_last_second(contact_log)
        min_cf = np.min(final_cf)
        self.assertTrue(min_cf >= 0.1,
                        "The average contact force dipped to %f during "
                        "the last second of simulation, while it should "
                        "have stayed above 0.1." % min_cf)
        
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
            "grasp_metrics.py"
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
        test_cases = [TestSetFive_GraspMetrics,
                      TestSetFive_ManipulationMpc]

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
