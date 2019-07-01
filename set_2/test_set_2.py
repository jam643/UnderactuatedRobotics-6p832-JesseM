import os
import imp
import sys
import timeout_decorator
import unittest
import numpy as np
from gradescope_utils.autograder_utils.decorators import weight
from gradescope_utils.autograder_utils.json_test_runner import JSONTestRunner

class TestSetTwo(unittest.TestCase):
    def setUp(self):
        pass

    @weight(2)
    @timeout_decorator.timeout(1.0)
    def test_problem_1_1(self):
        """Problem 1_1"""
        from set_2_for_testing import get_Q_f_problem_1_1
        Q_f = get_Q_f_problem_1_1()
        self.assertTrue(np.all(np.linalg.eigvals(Q_f) >= 0), "Q_f must be positive semidefinite")     
        self.assertEqual(Q_f.shape, (3,3), "Q_f must be 3x3")
        self.assertTrue(np.allclose(Q_f.transpose(), Q_f), "Q_f must be symmetric")
        self.assertLessEqual(np.sum(np.power(Q_f[:2,:],2)), 1e-6, "Q_f must not depend on x or y")

        # the below is a random test which tries to hide the answer to this question
        num_samples = 100
        random_states_batch = np.random.randn(3,num_samples)
        computed_final_costs = np.zeros(num_samples)
        for idx, val in enumerate(computed_final_costs):
            state = random_states_batch[:,idx]
            computed_final_costs[idx] = np.matmul(np.matmul(state.transpose(), Q_f),state)

        minimum_cost = np.argmin(computed_final_costs)

        deviations_from_0_theta = np.power(np.zeros(num_samples) - random_states_batch[2,:],2)
        minimum_deviation_from_0_theta = np.argmin(deviations_from_0_theta)
    
        self.assertEqual(minimum_cost, minimum_deviation_from_0_theta,
            "We produced a ton of random samples, and the one with smallest theta "
            "didn't have the smallest cost according to Q_f. Try again!")
        
    @weight(2)
    @timeout_decorator.timeout(1.0)
    def test_problem_1_2(self):
        """Problem 1_2"""
        from set_2_for_testing import get_Q_f_problem_1_2
        Q_f = get_Q_f_problem_1_2()

        self.assertTrue(np.all(np.linalg.eigvals(Q_f) >= 0), "Q_f must be positive semidefinite")     
        self.assertEqual(Q_f.shape, (3,3), "Q_f must be 3x3")
        self.assertTrue(np.allclose(Q_f.transpose(), Q_f), "Q_f must be symmetric")
        self.assertLessEqual(np.sum(np.power(Q_f[2,:],2)), 1e-6, "Q_f must not depend on yaw.")

        state = np.asarray([1,0.5])
        self.assertLessEqual(np.matmul(np.matmul(state.transpose(),Q_f[:2,:2]),state), 1e-6,
            "x=1, y=0.5 should have very low cost against Q_f, but they don't.")

        num_samples = 100
        random_states_batch = np.random.randn(3,num_samples)
        random_perturb = np.random.rand(1)

        # choose last point to be on line
        random_states_batch[:,-1] = [2*random_perturb,1*random_perturb,100]

        computed_final_costs = np.zeros(num_samples)
        for idx, val in enumerate(computed_final_costs):
            state = random_states_batch[:,idx]
            computed_final_costs[idx] = np.matmul(np.matmul(state.transpose(), Q_f),state)

        # the minimum should be the one specified to be on the line
        minimum_cost = np.argmin(computed_final_costs)
        self.assertEqual(minimum_cost, num_samples-1,
           "We produced a ton of random samples, and the one that was closest to "
           "the line y=0.5x didn't have the smallest cost according to Q_f. Try again!")

    @weight(2)
    @timeout_decorator.timeout(1.0)
    def test_problem_3_1_time_to_go(self):
        """Problem 3_1 Time (Cost) To Go"""
        from set_2_for_testing import get_optimal_time_to_go_problem_3_1 
        
        q_s = [5.0, -10.0, 0.0]
        qdot = 15.0

        self.assertEqual(get_optimal_time_to_go_problem_3_1(q=0, qdot=0), 0,
            "The optimal time to go at [q=0.0,qd=0.0] should be 0.")

        for q in q_s:
            self.assertAlmostEqual(abs(get_optimal_time_to_go_problem_3_1(q, qdot)),
                (qdot + 2*np.sqrt(0.5*qdot**2 + q)), delta=1e-6,
                msg="An optimal time-to-go test case for [q=%2.2f, qd=%2.2f] failed." % (q, qdot))

    @weight(2)
    @timeout_decorator.timeout(1.0)
    def test_problem_3_1_optimal_control(self):
        """Problem 3_1 Optimal Control Input"""
        from set_2_for_testing import get_optimal_control_problem_3_1

        self.assertEqual(get_optimal_control_problem_3_1(q=0, qdot=0), 0,
            "The optimal control at [q=0.0,qd=0.0] should be 0.")

        qdot = -10.0
        q_s = [5.0, -10.0, 0.0]
        for q in q_s:
            self.assertEqual(get_optimal_control_problem_3_1(q, qdot), 1,
                "Incorrect control applied at [q=%2.2f, qd=%2.2f]." % (q, qdot))

        qdot = 15.0
        for q in q_s:
            self.assertEqual(get_optimal_control_problem_3_1(q, qdot), -1,
                "Incorrect control applied at [q=%2.2f, qd=%2.2f]." % (q, qdot))

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
              notebook_file = "set_2_answers.ipynb",
              test_output_file = "test_results.json"):
    os.system("rm -f %s" % test_output_file)
    os.system("rm -f /tmp/set_2.py /tmp/set_2.pyc")
    os.system("jupyter nbconvert --ExecutePreprocessor.timeout=60 --output-dir /tmp --output set_2 --to python %s" % notebook_file)

    with open("/tmp/set_2.py") as f:
        content = f.readlines()
    filtered_content = []
    for i in content:
        if "matplotlib notebook" not in i:
            if "plt.show()" not in i:
                if 'test_set_2' not in i:
                    filtered_content.append(i)
    os.system("rm -f /tmp/set_2_for_testing.py /tmp/set_2_for_testing.pyc")
    os.system("touch /tmp/set_2_for_testing.py")
    with open('/tmp/set_2_for_testing.py', 'a') as the_file:
        for j in filtered_content:
            the_file.write(j+'\n')

    # Import this so it's in scope down the road...
    try:
        set_2_for_testing = timeout_decorator.timeout(60)(
            lambda: imp.load_source('set_2_for_testing', '/tmp/set_2_for_testing.py'))()
        do_testing = True

    except timeout_decorator.timeout_decorator.TimeoutError:
        global_fail_with_error_message("Timed out importing your notebook.", test_output_file)
        do_testing = False

    if do_testing:
        test_cases = [TestSetTwo]

        suite = unittest.TestSuite()
        for test_class in test_cases:
            tests = unittest.defaultTestLoader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

        with open(test_output_file, "w") as f:
            JSONTestRunner(stream=f).run(suite)

    os.system("rm -f /tmp/set_2_for_testing.py /tmp/set_2_for_testing.pyc")
    os.system("rm -f /tmp/set_2.py /tmp/set_2.pyc")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Please invoke with two arguments: the ipynb, and the results json to write."
        exit(1)

    run_tests(local=True, notebook_file=sys.argv[1], test_output_file=sys.argv[2])
