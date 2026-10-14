"""Regression tests for axis boundary saturation and recovery."""
import math
import unittest

import torch

from scene.gaussian_model import GaussianModel


def axis(tilt, side, length=1.0):
    tilt, side = math.radians(tilt), math.radians(side)
    return torch.tensor([math.sin(side), math.cos(side)*math.cos(tilt),
                         math.cos(side)*math.sin(tilt)], dtype=torch.float64)*length


class AxisProjectionTest(unittest.TestCase):
    def model(self, mode="project", cameras=1):
        model = GaussianModel(3, axis_mode="bounded_tilt", axis_constraint_mode=mode,
                              number_of_cameras=cameras, axis_tilt_init_deg=30,
                              axis_tilt_min_deg=29.5, axis_tilt_max_deg=30.5,
                              axis_side_init_deg=.2, axis_side_limit_deg=.5)
        model._axis = torch.nn.Parameter(axis(30, .2)[None].repeat(cameras,1))
        return model

    def assert_bounds(self, model):
        for i in range(model.number_of_cameras):
            result = model.get_axis(i).detach()
            tilt = math.degrees(math.atan2(result[2],result[1]))
            side = math.degrees(math.asin(result[0]))
            self.assertTrue(29.5-1e-8 <= tilt <= 30.5+1e-8)
            self.assertTrue(-.3-1e-8 <= side <= .7+1e-8)
            self.assertAlmostEqual(float(result.norm()), 1.0)

    def test_projection_preserves_feasible_axis_exactly(self):
        model = self.model()
        original = model._axis.detach().clone()
        model.project_axis_()
        self.assertTrue(torch.equal(original,model._axis))

    def test_both_bounds_and_cameras_preserve_vector_length(self):
        model = self.model(cameras=2)
        with torch.no_grad():
            model._axis.copy_(torch.stack([axis(34,-3,2),axis(27,3,4)]))
        lengths = model._axis.detach().norm(dim=1)
        model.project_axis_()
        self.assert_bounds(model)
        torch.testing.assert_close(model._axis.norm(dim=1),lengths)

    def test_projected_boundary_retains_inward_gradient(self):
        model = self.model()
        with torch.no_grad():model._axis.copy_(axis(34,-3)[None])
        model.project_axis_()
        loss = (model.get_axis(0)-axis(30,.2)).square().sum()
        loss.backward()
        self.assertGreater(float(model._axis.grad.norm()),0)
        before = float(loss.detach())
        optimizer = torch.optim.SGD([model._axis],lr=.1)
        optimizer.step();model.project_axis_()
        self.assertLess(float((model.get_axis(0)-axis(30,.2)).square().sum().detach()),before)
        self.assert_bounds(model)

    def test_adam_recovers_after_outward_momentum(self):
        model = self.model()
        optimizer = torch.optim.Adam([model._axis],lr=.005)
        for target, steps in [(axis(34,-3),60),(axis(30,.2),150)]:
            for _ in range(steps):
                optimizer.zero_grad()
                (model.get_axis(0)-target).square().sum().backward()
                optimizer.step();model.project_axis_();self.assert_bounds(model)
            if steps==60:
                stuck = model.get_axis(0).detach().clone()
        self.assertGreater(float((model.get_axis(0)-stuck).norm().detach()),.001)
        self.assertLess(float((model.get_axis(0)-axis(30,.2)).norm().detach()),.001)

    def test_legacy_saturation_remains_available_as_control(self):
        model = self.model("clamp")
        with torch.no_grad():model._axis.copy_(axis(34,-3)[None])
        original = model._axis.detach().clone()
        model.project_axis_()
        self.assertTrue(torch.equal(original,model._axis))
        (model.get_axis(0)-axis(30,.2)).square().sum().backward()
        self.assertEqual(float(model._axis.grad.norm()),0.0)

    def test_free_axis_is_not_projected(self):
        model = self.model();model.axis_mode="free"
        with torch.no_grad():model._axis.copy_(axis(100,10)[None])
        original = model._axis.detach().clone()
        model.project_axis_()
        self.assertTrue(torch.equal(original,model._axis))


if __name__ == '__main__':unittest.main()
