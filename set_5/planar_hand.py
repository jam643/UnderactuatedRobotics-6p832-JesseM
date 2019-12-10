# -*- coding: utf8 -*-

import argparse
import math
import functools
import os.path
import time
import random
import sys

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import scipy as sp

from pydrake.all import (DirectCollocation, FloatingBaseType,
                         PiecewisePolynomial, RigidBodyTree, RigidBodyPlant,
                         SolutionResult, VectorSystem, RigidBodyFrame,
                         LeafSystem, PortDataType, BasicVector,
                         MathematicalProgram, Shape,
                         DiagramBuilder, FloatingBaseType, Simulator,
                         SignalLogger, CompliantMaterial,
                         AddModelInstancesFromSdfString, Solve)
from pydrake.systems.framework import AbstractValue
from pydrake.forwarddiff import jacobian
from pydrake.solvers import ik
from underactuated import (PlanarRigidBodyVisualizer)
from pydrake.multibody.plant import ContactResults
#import pydrake.attic.multibody.rigid_body_plant.ContactResults as ContactResults
import pydrake.attic.multibody.rigid_body_plant as mut

from grasp_metrics import achieves_force_closure
from grasp_metrics import compute_convex_hull_volume


class HandController(LeafSystem):
    ''' This system consumes state from a simulation
        of a multifingered hand with a single unactuated
        manipuland in its grip, and produces torque
        commands for each joint of the hand. '''
    
    ''' 
    The class constructor sets up the input and output
    information of the system, and also searches for
    grasp points to use on the object.
    Arguments:
    - hand: The hand RigidBodyTree
    - x_nom: The nominal posture of the hand.
    - num_fingers: The number of fingers used when constructing
        the hand.
    - manipuland_trajectory_callback: a function accepting one
        floating point number (the time t), and returning a
        3-length numpy vector representing the desired manipuland
        [x, y, theta] at time t. A default callback is used
        if None is supplied.
    - manipuland_link_name: What link of the robot is the manipuland?
    - finger_link_name: The name of the fingertip link of
        each finger.
    - mu: The coefficient of friction assumed by the controller.
    - n_grasp_search_iters: How many random samples of grasps to
        produce while searching for a grasp.
    - control_period: The update rate of the controller. This
        system will produce output as quickly as desired,
        but will only compute new control actions at this rate.
    - print_period: The controller will print to the terminal
        at this period (in simulation time) as a progress update.
    '''
    def __init__(self, hand, x_nom, num_fingers,
                 manipuland_trajectory_callback = None,
                 manipuland_link_name="manipuland_body",
                 finger_link_name="link_3",
                 mu=0.5,
                 n_grasp_search_iters=200,
                 control_period = 0.0333,
                 print_period=1.0):
        LeafSystem.__init__(self)
        
        # Copy lots of stuff
        self.hand = hand
        self.x_nom = x_nom
        self.nq = hand.get_num_positions()
        self.nu = hand.get_num_actuators()
        self.manipuland_link_name = manipuland_link_name
        self.manipuland_trajectory_callback = manipuland_trajectory_callback
        self.n_grasp_search_iters = n_grasp_search_iters
        self.mu = mu
        self.num_fingers = num_fingers
        self.num_finger_links = (self.nq - 3) / num_fingers
        self.fingertip_position = np.array([1.2, 0., 0.])
        self.print_period = print_period
        self.last_print_time = -print_period
        self.shut_up = False

        self.DeclareInputPort(PortDataType.kVectorValued,
                               hand.get_num_positions() +
                               hand.get_num_velocities())

        self.DeclareDiscreteState(self.nu)
        self.DeclarePeriodicDiscreteUpdate(period_sec=control_period)

        hand_actuators = hand.get_num_actuators()
        self.finger_link_indices = []
        # The hand plant wants every finger input as a
        # separate vector. Rather than using a mux down the
        # road, we'll just output in the format it wants.
        for i in range(num_fingers):
            self.DeclareVectorOutputPort(
                BasicVector(hand_actuators / num_fingers),
                functools.partial(self.DoCalcVectorOutput, i))
            self.finger_link_indices.append(
                self.hand.FindBody("link_3", model_id=i).get_body_index()
            )

        # Generate manipuland planar geometry and sample grasps.
        self.PlanGraspPoints()

    '''
    Extracts manipuland geometry in the plane of the simulation
    (XY plane), and then randomly samples grasps over the
    geometry, querying out to the grasp evaluation metrics you
    wrote for the other half of this assignment.
    
    A given randomly sampled set of points on the object
    surface are generated such that no pair of them is
    closer than the closest distance two fingertips will
    fit together. Once the grasp is verified to be have
    force closure, the volume of the convex hull of its
    grasp points is compared to the last best grasp, and if
    this grasp has smaller volume, this grasp is ignored.
    Fingertips are then matched to candidate grasp
    points in a set (in a one-to-one matching) by an LP that
    minimizes the total Euclidean distance between fingertip
    position in the hand nominal posture, and each candidate
    grasp position on the object in its starting position.
    The grasp candidate is finally sanity-checked by running IK to
    ensure that the hand can reach all of the points,
    and accepted as the new best candidate grasp if all of these
    things work in order.
    
    Aren't you glad you didn't have to write this part :)
    '''
    def PlanGraspPoints(self):
        # First, extract the bounding geometry of the object.
        # Generally, our geometry is coming from 3d models, so
        # we have to do some legwork to extract 2D geometry. For
        # the examples we'll use in this set, we'll assume
        # that extracting the convex hull of the first visual element
        # is a good representation of the object geometry. (This is
        # not great! But it'll do the job for us, since we're going
        # to experiment with only simple objects.)
        kinsol = self.hand.doKinematics(self.x_nom[0:self.hand.get_num_positions()])
        self.manipuland_link_index = \
            self.hand.FindBody(self.manipuland_link_name).get_body_index()
        body = self.hand.get_body(self.manipuland_link_index)
        # For projecting into XY plane
        Tview = np.array([[1., 0., 0., 0.],
                          [0., 1., 0., 0.],
                          [0., 0., 0., 1.]])
        all_points = ExtractPlanarObjectGeometryConvexHull(body, Tview)

        # For later use: precompute the fingertip positions of the
        # robot in the nominal posture.
        nominal_fingertip_points = np.empty((2, self.num_fingers))
        for i in range(self.num_fingers):
            nominal_fingertip_points[:, i] = self.hand.transformPoints(
                kinsol, self.fingertip_position, 
                self.finger_link_indices[i], 0)[0:2, 0]

        # Search for an optimal grasp with N points
        random.seed(42)
        np.random.seed(42)

        def get_random_point_and_normal_on_surface(all_points):
            num_points = all_points.shape[1]
            i = random.choice(range(num_points-1))
            first_point = np.asarray([all_points[0][i], all_points[1][i]])
            second_point = np.asarray([all_points[0][i+1], all_points[1][i+1]])
            first_to_second = second_point - first_point
            # Clip to avoid interpolating close to a corner
            interpolate_param = np.clip(np.random.rand(), 0.2, 0.8)
            rand_point = first_point + interpolate_param*first_to_second

            normal = np.array([-first_to_second[1], first_to_second[0]]) \
                / np.linalg.norm(first_to_second)
            return rand_point, normal

        best_conv_volume = 0
        best_points = []
        best_normals = []
        best_finger_assignments = []
        for i in range(self.n_grasp_search_iters):
            grasp_points = []
            normals = []
            for j in range(self.num_fingers):
                point, normal = \
                    get_random_point_and_normal_on_surface(all_points)
                # check for duplicates or close points -- fingertip
                # radius is 0.2, so require twice that margin to avoid
                # intersection fingertips.
                num_rejected = 0
                while min([1.0] + [np.linalg.norm(grasp_point - point, 2)
                                   for grasp_point in grasp_points]) <= 0.4:
                    point, normal = \
                        get_random_point_and_normal_on_surface(all_points)
                    num_rejected += 1
                    if num_rejected > 10000:
                        print "Rejected 10000 points in a row due to crowding." \
                              " Your object is a bit too small for your number of" \
                              " fingers."
                        break
                grasp_points.append(point)
                normals.append(normal)
            if achieves_force_closure(grasp_points, normals, self.mu):
                # Test #1: Rank in terms of convex hull volume of grasp points
                volume = compute_convex_hull_volume(grasp_points)
                if volume < best_conv_volume:
                    continue
                    
                # Test #2: Does IK work for this point?
                self.grasp_points = grasp_points
                self.grasp_normals = normals

                # Pick optimal finger assignment that
                # minimizes distance between fingertips in the
                # nominal posture, and the chosen grasp points
                grasp_points_world = self.transform_grasp_points_manipuland(
                    self.x_nom)[0:2, :]

                prog = MathematicalProgram()
                # We'd *really* like these to be binary variables,
                # but unfortunately don't have a MIP solver available in the
                # course docker container. Instead, we'll solve an LP,
                # and round the result to the nearest feasible integer
                # solutions. Intuitively, this LP should probably reach its
                # optimal value at an extreme point (where the variables
                # all take integer value). I've not yet observed this
                # not occuring in practice!
                assignment_vars = prog.NewContinuousVariables(
                    self.num_fingers, self.num_fingers, "assignment")
                for grasp_i in range(self.num_fingers):
                    # Every row and column of assignment vars sum to one --
                    # each finger matches to one grasp position
                    prog.AddLinearConstraint(
                        np.sum(assignment_vars[:, grasp_i]) == 1.)
                    prog.AddLinearConstraint(
                        np.sum(assignment_vars[grasp_i, :]) == 1.)
                    for finger_i in range(self.num_fingers):
                        # If this grasp assignment is active,
                        # add a cost equal to the distance between
                        # nominal pose and grasp position
                        prog.AddLinearCost(
                            assignment_vars[grasp_i, finger_i] *
                            np.linalg.norm(
                                grasp_points_world[:, grasp_i] -
                                nominal_fingertip_points[:, finger_i]))
                        prog.AddBoundingBoxConstraint(
                            0., 1., assignment_vars[grasp_i, finger_i])
                result = Solve(prog)
                assignments = result.GetSolution(assignment_vars)
                # Round assignments to nearest feasible set
                claimed = [False]*self.num_fingers
                for grasp_i in range(self.num_fingers):
                    order = np.argsort(assignments[grasp_i, :])
                    fill_i = self.num_fingers - 1
                    while claimed[order[fill_i]] is not False:
                        fill_i -= 1
                    if fill_i < 0:
                        raise Exception("Finger association failed. "
                                        "Horrible bug. Tell Greg")
                    assignments[grasp_i, :] *= 0.
                    assignments[grasp_i, order[fill_i]] = 1.
                    claimed[order[fill_i]] = True

                # Populate actual assignments
                self.grasp_finger_assignments = []
                for grasp_i in range(self.num_fingers):
                    for finger_i in range(self.num_fingers):
                        if assignments[grasp_i, finger_i] == 1.:
                            self.grasp_finger_assignments.append(
                                (finger_i, 
                                 np.array(self.fingertip_position)))
                            continue

                qinit, info = self.ComputeTargetPosture(
                                self.x_nom, self.x_nom[(self.nq-3):self.nq],
                                backoff_distance=0.2)
                if info != 1:
                    continue

                best_conv_volume = volume
                best_points = grasp_points
                best_normals = normals
                best_finger_assignments = self.grasp_finger_assignments

        if len(best_points) == 0:
            print "After %d attempts, couldn't find a good grasp "\
                  "for this object." % self.n_grasp_search_iters
            print "Proceeding with a horrible random guess."
            best_points = [np.random.uniform(-1., 1., 2)
                           for i in range(self.num_fingers)]
            best_normals = [np.random.uniform(-1., 1., 2)
                            for i in range(self.num_fingers)]
            best_finger_assignments = [(i, self.fingertip_position)
                                       for i in range(self.num_fingers)]

        self.grasp_points = best_points
        self.grasp_normals = best_normals
        self.grasp_finger_assignments = best_finger_assignments

    ''' Shells out to the manipuland object position callback
        if available, or provides a default implementation if not. '''
    def GetDesiredObjectPosition(self, t):
        if self.manipuland_trajectory_callback is not None:
            return self.manipuland_trajectory_callback(t)
        else:
            return np.array([1.0 + 0.5*np.cos(t), 0.5*np.sin(t), 0.5*np.sin(t)])

    ''' Sets up and solves an inverse kinematics (IK) problem as
        a nonlinear optimization. Searches for a posture of the
        hand that is closest to the nominal pose (x_nom) of the
        hand, subject to the constraints:
            1) The manipuland must be in the target pose.
            2) The fingertip positions must be very close to
              the grasp points on the manipuland surface.
            3) The fingertips must point within 1 radian of the
              graso normals on the manipuland surface.
            4) The hand joint angles must be within [-pi, pi].
        This method assumes the grasp points have already been
        computed (which happens in the class constructor). 
        
        This optimization is seeded with the configuration x_seed,
        so results are likely to be close to that point, as this
        is a nonlinear optimization.
        
        The optional argument backoff_distance changes constraint
        (2) to instead constrain the fingertips to be very
        close to a point backoff_distance away from the grasp point,
        along the grasp normal direction. (This could be used
        to start the simulation in a pose just before contact
        has occured.)'''
    def ComputeTargetPosture(self, x_seed, target_manipuland_pose,
                             backoff_distance=0.0):
        q_des_full = np.zeros(self.nq)
        q_des_full[self.nq-3:] = target_manipuland_pose

        desired_positions = self.transform_grasp_points_manipuland(q_des_full)
        desired_normals = self.transform_grasp_normals_manipuland(q_des_full)
        desired_positions -= backoff_distance * desired_normals  # back off a bit
        desired_angles = [
            math.atan2(desired_normals[1, k], desired_normals[0, k])
            for k in range(desired_normals.shape[1])
        ]
        constraints = []

        # Constrain the fingertips to lie on the grasp points,
        # facing approximately along the grasp angles.
        for i in range(len(self.grasp_points)):
            constraints.append(
                ik.WorldPositionConstraint(
                    self.hand,
                    self.finger_link_indices[
                        self.grasp_finger_assignments[i][0]],
                    self.grasp_finger_assignments[i][1],
                    desired_positions[:, i]-0.01,  # lb
                    desired_positions[:, i]+0.01)  # ub
            )

            constraints.append(
                ik.WorldEulerConstraint(
                    self.hand,
                    self.finger_link_indices[self.grasp_finger_assignments[i][0]],
                    [-0.01, -0.01, desired_angles[i]-1.0],  # lb
                    [0.01, 0.01, desired_angles[i]+1.0])  # ub
            )

        posture_constraint = ik.PostureConstraint(self.hand)
        # Disambiguate the continuous rotation joint angle choices
        # for the hand.
        posture_constraint.setJointLimits(
            np.arange(self.nq-3),
            -np.ones(self.nq-3)*math.pi,
            np.ones(self.nq-3)*math.pi)
        # Constrain the manipuland to be in the target position.
        posture_constraint.setJointLimits(
            [self.nq-3, self.nq-2, self.nq-1],
            target_manipuland_pose-0.01, target_manipuland_pose+0.01)
        constraints.append(posture_constraint)

        options = ik.IKoptions(self.hand)
        results = ik.InverseKin(
            self.hand, x_seed[0:self.nq], self.x_nom[0:self.nq], constraints, options)
        return results.q_sol[0], results.info[0]

    ''' Helper to transform the chosen grasp points
        into the manipuland frame when the full robot
        state is x. '''
    def transform_grasp_points_manipuland(self, x):
        kinsol = self.hand.doKinematics(x[0:self.hand.get_num_positions()])
        points = np.vstack(self.grasp_points).T
        points = np.vstack([points, np.zeros(len(self.grasp_points))])

        return self.hand.transformPoints(
            kinsol, points, self.manipuland_link_index, 0)

    ''' Helper to transform the chosen grasp normals
        into the manipuland frame when the full robot
        state is x. '''
    def transform_grasp_normals_manipuland(self, x):
        kinsol = self.hand.doKinematics(x[0:self.hand.get_num_positions()])
        points = np.vstack(self.grasp_normals).T
        points = np.vstack([points, np.zeros(len(self.grasp_normals))])

        tf = self.hand.relativeTransform(kinsol, self.manipuland_link_index, 0)
        return np.dot(tf[0:3, 0:3].T, points)

    ''' Helper to produce a list of points in world frame
        of the fingertip positions, in grasp point order (!),
        when the full robot state is x. '''
    def transform_grasp_points_fingers(self, x):
        kinsol = self.hand.doKinematics(x[0:self.hand.get_num_positions()])
        points = np.empty((3, len(self.grasp_finger_assignments)),
                          dtype=x.dtype)
        for i, gfa in enumerate(self.grasp_finger_assignments):
            points[:, i] = self.hand.transformPoints(
                kinsol, gfa[1], self.finger_link_indices[gfa[0]], 0)[:, 0]
        return points

    ''' Needs to implement a controller following the spec in
        set_5_mpc.ipynb.
        
        Inputs:
        x: The current state of the robot.
        t: The current time.
        
        Outputs:
        A 1-d numpy array of length self.nq joint torques
        for the hand. '''
    def ComputeControlInput(self, x, t):
        # Set up things you might want...
        q = x[0:self.nq]
        v = x[self.nq:]

        kinsol = self.hand.doKinematics(x[0:self.hand.get_num_positions()])
        M = self.hand.massMatrix(kinsol)
        C = self.hand.dynamicsBiasTerm(kinsol, {}, None)
        B = self.hand.B

        # Assume grasp points are achieved, and calculate
        # contact jacobian at those points, as well as the current
        # grasp points and normals (in WORLD FRAME).
        grasp_points_world_now = self.transform_grasp_points_manipuland(q)
        grasp_normals_world_now = self.transform_grasp_normals_manipuland(q)
        J_manipuland = jacobian(
            self.transform_grasp_points_manipuland, q)
        ee_points_now = self.transform_grasp_points_fingers(q)
        J_manipulator = jacobian(
            self.transform_grasp_points_fingers, q)

        # The contact jacobian (Phi), for contact forces coming from
        # point contacts, describes how the movement of the contact point
        # maps into the joint coordinates of the robot. We can get this
        # by combining the jacobians derived from forward kinematics to each
        # of the two bodies that are in contact, evaluated at the contact point
        # in each body's frame.
        J_contact = J_manipuland - J_manipulator
        # Cut off the 3rd dimension, which should always be 0
        J_contact = J_contact[0:2, :, :]
        # Given these, the manipulator equations enter as a linear constraint
        # M qdd + C = Bu + J_contact lambda
        # (the combined effects of lambda on the robot).
        # The list of grasp points, in manipuland frame, that we'll use.
        n_cf = len(self.grasp_points)
        # The evaluated desired manipuland posture.
        manipuland_qdes = self.GetDesiredObjectPosition(t)

        # The desired robot (arm) posture, calculated via inverse kinematics
        # to achieve the desired grasp points on the object in its current
        # target posture.
        qdes, info = self.ComputeTargetPosture(x, manipuland_qdes)
        if info != 1:
            if not self.shut_up:
                print "Warning: target posture IK solve got info %d " \
                      "when computing goal posture at least once during " \
                      "simulation. This means the grasp points was hard to " \
                      "achieve given the current object posture. This is " \
                      "occasionally OK, but indicates that your controller " \
                      "is probably struggling a little." % info
                self.shut_up = True

        # From here, it's up to you. Following the guidelines in the problem
        # set, implement a controller to achieve the specified goals.

        '''
        YOUR CODE HERE
        '''
        u = np.zeros(self.nu)
        return u

    ''' This is called on every discrete state update (at the
        specified control period), and expects the discrete state
        to be updated with the new discrete state after the update.
        
        For this system, the state is the output we'd like to produce
        (for the complete robot). This system could be created
        in a stateless way pretty easily -- through a combination of
        limiting the publish rate of the system, and adding multiplexers
        after the system to split the output into the
        individual-finger-sized chunks that the hand plant wants
        to consume. '''
    def DoCalcDiscreteVariableUpdates(self, context, events, discrete_state):
        # Call base method to ensure we do not get recursion.
        LeafSystem.DoCalcDiscreteVariableUpdates(
            self, context, events, discrete_state)

        new_control_input = discrete_state. \
            get_mutable_vector().get_mutable_value()
        x = self.EvalVectorInput(context, 0).get_value()
        old_u = discrete_state.get_mutable_vector().get_mutable_value()
        new_u = self.ComputeControlInput(x, context.get_time())
        new_control_input[:] = new_u

    ''' This is called whenever this system needs to publish
        output. We did some magic in the constructor to add
        an extra argument to tell the function what finger's
        control input to return. It looks up into the
        current state what the current complete output is,
        and returns the torques for only finger i.'''
    def DoCalcVectorOutput(self, i, context, y_data):
        if (self.print_period and
                context.get_time() - self.last_print_time
                >= self.print_period):
            print "t: ", context.get_time()
            self.last_print_time = context.get_time()
        control_output = context.get_discrete_state_vector().get_value()
        y = y_data.get_mutable_value()
        # Get the ith finger control output
        y[:] = control_output[(i*self.num_finger_links):((i+1)*self.num_finger_links)]


def ExtractPlanarObjectGeometryConvexHull(body, Tview):
    ''' Given a RigidBody object, returns a numpy array
        of points (shape 2xN) in counterclockwise order
        around the convex hull of the object's first
        visual geometry, as seen via projection matrix Tview.

        This function is a convenience function for this
        problem set, but probably doesn't do exactly what
        you want it to for complex shapes. '''

    # (TODO:gizatt Switch to collision geometry when the
    # appropriate bindings are in.)
    elements = body.get_visual_elements()

    if len(elements) > 1:
        print "Warning: ignoring everything but the first " \
              "visual element in the manipuland model."

    element = elements[0]

    if not element.hasGeometry():
        raise ValueException("Visual element for manipuland had no geometry!")

    element_local_tf = element.getLocalTransform()
    geom = element.getGeometry()
    # Prefer to use faces when possible.
    if geom.hasFaces():
        try:
            points = geom.getPoints()
            # Unnecessary if we're taking a convex hull
            # tris = geom.getFaces()
            # for tri in tris:
            #    new_pts = np.transpose(np.vstack([points[:, x]
            #                           for x in tri]))
            #    patch.append(new_pts)
            patch = points
        except Exception as e:
            print "Exception when loading tris from " \
                  "geometry: ", e
    else:
        geom_type = geom.getShape()
        if geom_type == Shape.SPHERE:
            # Sphere will return their center as their
            # getPoint(). So generate a better patch by
            # sampling points around the sphere surface
            # in the view plane.
            center = geom.getPoints()
            sample_pts = np.arange(0., 2.*math.pi, 0.25)
            patch = np.vstack([math.cos(pt)*Tview[0, 0:3]
                               + math.sin(pt)*Tview[1, 0:3]  # noqa
                               for pt in sample_pts])
            patch = np.transpose(patch)
            patch *= geom.radius
        else:
            # Other geometry types will usually return their
            # bounding box, which is good enough for basic
            # visualization. (This could be improved for
            # capsules?)
            patch = geom.getPoints()

    # Convert to homogenous coords and move out of body frame
    patch = np.vstack((patch, np.ones((1, patch.shape[1]))))
    patch = np.dot(element_local_tf, patch)
    patch = np.dot(Tview, patch)

    # Take convhull of resulting points
    if patch.shape[1] > 3:
        hull = sp.spatial.ConvexHull(
            np.transpose(patch[0:2, :]))
        patch = np.transpose(
            np.vstack([patch[:, v] for v in hull.vertices]))

    # Close path if not closed
    if (patch[:, -1] != patch[:, 0]).any():
        patch = np.hstack((patch, patch[:, 0][np.newaxis].T))

    return patch[0:2, :]


class PlanarHandContactLogger(LeafSystem):
    ''' Logs contact force history, using
        the planar hand plant contact result
        output port.
        
        Stores sample times, accessible via
        sample_times(), and contact results for
        each sample time, accessible as a list
        from method data().
        
        Every contact result is a list of tuples,
        one tuple for each contact,
        where each tuple contains (id_1, id_2, r, f):
            id_1 = the ID of element #1 in collision
            id_2 = the ID of element #2 in collision
            r = the contact location, in world frame
            f = the contact force, in world frame'''
    def __init__(self,
                 hand_controller,
                 hand_plant):
        LeafSystem.__init__(self)
        self.hand_controller = hand_controller
        self.hand_plant = hand_plant
        
        self.n_cf = len(hand_controller.grasp_points)
        self._data = []
        self._sample_times = np.empty((0, 1))
        self.shut_up = False
        # Contact results
        # self.DeclareInputPort('contact_results', PortDataType.kAbstractValued,
        #                        hand_plant.contact_results_output_port().size())
        self.DeclareAbstractInputPort(
            "contact_results", AbstractValue.Make(mut.ContactResults()))

    def data(self):
        return self._data
   
    def sample_times(self):
        return self._sample_times
    
    def DoPublish(self, context, events):
        contact_results = self.EvalAbstractInput(context, 0).get_value()
        self._sample_times = np.vstack([self._sample_times, [context.get_time()]])
        
        this_contact_info = []
        for contact_i in range(contact_results.get_num_contacts()):
            if contact_i >= self.n_cf:
                if not self.shut_up:
                    print "More contacts than expected (the # of grasp points). "\
                          "Dropping some! Your fingertips probably touched each other."
                    self.shut_up = True
                break
            # Cludgy -- would rather keep things as objects.
            # But I need to work out how to deepcopy those objects.
            # (Need to bind their various constructive methods)
            contact_info = contact_results.get_contact_info(contact_i)
            contact_force = contact_info.get_resultant_force()
            this_contact_info.append([
                contact_info.get_element_id_1(),
                contact_info.get_element_id_2(),
                contact_force.get_application_point(),
                contact_force.get_force()
            ])
        self._data.append(this_contact_info)
        

class PlanarHandExtrasVisualizer():
    ''' Helps visualize a planar hand simulation.
        Cooperates with PlanarRigidBodyVisualizer,
        layering contact force visualization on
        top of the regular PRBV view.

        Assumes the PlanarHand object is already
        initialized (so we can extract its contact
        point information.)'''
    def __init__(self,
                 hand_controller,
                 hand_plant,
                 prbv,
                 fig,
                 ax,
                 show_forces=True):
        self.hand_controller = hand_controller
        self.hand_plant = hand_plant
        self.prbv = prbv
        self.fig = fig
        self.ax = ax
        self.show_forces = show_forces

        # Use a small number of quiver arrows to indicate
        # contact forces.
        self.n_cf = len(hand_controller.grasp_points)
        # Creates an appropriate number of zero-length arrows.
        self.Q = ax.quiver(np.zeros(self.n_cf),
                           np.zeros(self.n_cf),
                           np.zeros(self.n_cf),
                           np.zeros(self.n_cf),
                           pivot='tail',
                           color='b',
                           units='xy',
                           scale=2.0)

    def draw(self, x, contact_results):
        ''' Evaluates the robot state and draws it.
            Expects a raw state vector, not a context.'''
        if self.show_forces:
            tree = self.hand_plant.get_rigid_body_tree()
            positions = x[0:tree.get_num_positions()]
            kinsol = tree.doKinematics(positions)
            #print "NEW UPDATE "
            new_X = np.ones(self.n_cf)*100 # guaranteed off-screen
            new_Y = np.ones(self.n_cf)*100
            new_U = np.zeros(self.n_cf)
            new_V = np.zeros(self.n_cf)
            for i, contact_info in enumerate(contact_results):
                id_1, id_2, r, f = contact_info
                new_X[i] = r[0]
                new_Y[i] = r[1]
                new_U[i] = -f[0]
                new_V[i] = -f[1]
            self.Q.set_offsets(np.vstack([new_X, new_Y]).T)
            self.Q.set_UVC(new_U, new_V)

    def animate(self, state_log, contact_log, timestep, repeat=False):
        t = state_log.sample_times()
        x = state_log.data()

        # Contact log is hard to resample in time, but the
        # state log is much easier
        import scipy.interpolate
        t_resample = contact_log.sample_times()
        x = scipy.interpolate.interp1d(t, x, kind='linear', axis=1)(t_resample)
        t = t_resample

        def animate_update(i):
            self.prbv.draw(x[:, i])
            #self.draw(x[:, i], None)
            self.draw(x[:, i], contact_log.data()[i])

        ani = animation.FuncAnimation(self.fig,
                                      animate_update,
                                      t.shape[0],
                                      interval=1000*timestep,
                                      repeat=repeat)
        return ani


def BuildHand(num_fingers=3,
              manipuland_sdf="models/manipuland_box.sdf"):
    ''' Build up the hand by replicating a finger
        model at a handful of base positions. '''
    finger_base_positions = np.vstack([
         np.abs(np.linspace(-0.25, 0.25, num_fingers)),
         np.linspace(0.25, -0.25, num_fingers),
         np.zeros(num_fingers)]).T

    tree = RigidBodyTree()
    for i, base_pos in enumerate(finger_base_positions):
        frame = RigidBodyFrame(
            name="finger_%d_base_frame" % i,
            body=tree.world(),
            xyz=base_pos,
            rpy=[0, 0, 0])
        tree.addFrame(frame)
        AddModelInstancesFromSdfString(
            open("models/planar_finger.sdf", 'r').read(),
            FloatingBaseType.kFixed,
            frame, tree)

    # Add the manipuland as well from an sdf.
    manipuland_frame = RigidBodyFrame(
        name="manipuland_frame",
        body=tree.world(),
        xyz=[0, 0., 0.],
        rpy=[0, 0, 0])
    tree.addFrame(manipuland_frame)
    AddModelInstancesFromSdfString(
        open(manipuland_sdf, 'r').read(),
        FloatingBaseType.kFixed,
        manipuland_frame, tree)
    
    return tree


def SimulateHand(duration=10., 
                 num_fingers=3,
                 mu=0.5,
                 manipuland_sdf="models/manipuland_box.sdf",
                 initial_manipuland_pose=np.array([1.5, 0., 0.]),
                 n_grasp_search_iters=100,
                 manipuland_trajectory_callback=None,
                 control_period=0.0333,
                 print_period=1.0):
    ''' Given a great many passthrough arguments
        (see docs for HandController and
        usage example in set_5_mpc.ipynb), constructs
        a simulation of a num_fingers-fingered hand
        and simulates it for duration seconds from
        a specified initial manipuland pose. 
        
        Returns:
        (hand, plant, controller, state_log, contact_log)
        hand: The RigidBodyTree of the complete hand.
        plant: The RigidBodyPlant that owns the hand RBT
        controller: The HandController object
        state_log: A SignalLogger that has logged the output
        of the state output port of plant.
        contact_log: A PlanarHandContactLogger object that
        has logged the output of the contact results output
        port of the plant. '''
    builder = DiagramBuilder()

    tree = BuildHand(num_fingers, manipuland_sdf)
    num_finger_links = 3 # from sdf
    num_hand_q = num_finger_links * num_fingers
    
    # Generate the nominal posture for the hand
    # First link wide open, next links at right angles
    x_nom = np.zeros(2*num_finger_links*num_fingers + 6)
    for i in range(num_fingers):
        if i < num_fingers/2:
            x_nom[(num_finger_links*i):(num_finger_links*(i+1))] = \
                np.array([2, 1.57, 1.57])
        else:
            x_nom[(num_finger_links*i):(num_finger_links*(i+1))] = \
                -np.array([2, 1.57, 1.57])

    # Drop in the initial manipuland location
    x_nom[num_hand_q:(num_hand_q+3)] = initial_manipuland_pose
        

    # A RigidBodyPlant wraps a RigidBodyTree to allow
    # forward dynamical simulation. It handles e.g. collision
    # modeling.
    plant = builder.AddSystem(RigidBodyPlant(tree))
    # Alter the ground material used in simulation to make
    # it dissipate more energy (to make the object less bouncy)
    # and stickier (to make it easier to hold) and softer
    # (again to make it less bouncy)
    allmaterials = CompliantMaterial()
    allmaterials.set_youngs_modulus(1E6)  # default 1E9
    allmaterials.set_dissipation(1.0)     # default 0.32
    allmaterials.set_friction(0.9)        # default 0.9.
    plant.set_default_compliant_material(allmaterials)

    # Spawn a controller and hook it up
    controller = builder.AddSystem(
        HandController(
            tree,
            x_nom=x_nom,
            num_fingers=num_fingers,
            mu=mu,
            n_grasp_search_iters=n_grasp_search_iters,
            manipuland_trajectory_callback = manipuland_trajectory_callback,
            control_period=control_period,
            print_period=print_period))

    nq = controller.nq
    qinit, info = controller.ComputeTargetPosture(x_nom, x_nom[(nq-3):nq],
                                                  backoff_distance=0.0)
    if info != 1:
        print "Warning: initial condition IK solve returned info ", info
    xinit = np.zeros(x_nom.shape)
    xinit[0:(nq-3)] = qinit[0:-3]
    xinit[(nq-3):nq] = x_nom[(nq-3):nq]
    builder.Connect(plant.get_output_port(0), controller.get_input_port(0))
    for i in range(num_fingers):
        builder.Connect(controller.get_output_port(i), plant.get_input_port(i))

    # Create a state logger to log at 30hz
    state_log = builder.AddSystem(SignalLogger(plant.get_num_states()))
    state_log.DeclarePeriodicPublish(0.0333, 0.0)  # 30hz logging
    builder.Connect(plant.get_output_port(0), state_log.get_input_port(0))

    # And a contact result logger, same rate
    contact_log = builder.AddSystem(PlanarHandContactLogger(controller, plant))
    contact_log.DeclarePeriodicPublish(0.0333, 0.0)
    builder.Connect(plant.contact_results_output_port(),
                    contact_log.get_input_port(0))

    # Create a simulator
    diagram = builder.Build()
    simulator = Simulator(diagram)
    # Don't limit realtime rate for this sim, since we
    # produce a video to render it after simulating the whole thing.
    # simulator.set_target_realtime_rate(100.0)
    simulator.set_publish_every_time_step(False)

    # Force the simulator to use a fixed-step integrator,
    # which is much faster for this stiff system. (Due to the
    # spring-model of collision, the default variable-timestep
    # integrator will take very short steps. I've chosen the step
    # size here to be fast while still being stable in most situations.)
    integrator = simulator.get_mutable_integrator()
    integrator.set_fixed_step_mode(True)
    integrator.set_maximum_step_size(0.005)

    # Set the initial state
    state = simulator.get_mutable_context().\
        get_mutable_continuous_state_vector()
    state.SetFromVector(xinit)

    # Simulate!
    simulator.StepTo(duration)

    return tree, plant, controller, state_log, contact_log
