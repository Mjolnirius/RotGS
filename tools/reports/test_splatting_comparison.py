import math
import unittest

import numpy as np

from tools.reports.build_splatting_comparison import fit_camera_ring, normalized_coordinates


class ComparisonFrameTests(unittest.TestCase):
    def test_camera_ring_recovers_rotated_translated_axis_without_tae(self):
        z=np.array([.3,.4,math.sqrt(.75)])
        x=np.cross(z,[0.,1.,0.]);x/=np.linalg.norm(x)
        y=np.cross(z,x)
        center=np.array([10.,-4.,2.])
        cameras=[]
        for theta in np.linspace(0,2*np.pi,72,endpoint=False):
            position=center+3*(np.cos(theta)*x+np.sin(theta)*y)
            rotation=np.column_stack((x,-z,y))
            cameras.append(dict(position=position,rotation=rotation))
        recovered,vertical,front=fit_camera_ring(cameras)
        np.testing.assert_allclose(recovered,center,atol=1e-12)
        np.testing.assert_allclose(vertical,z,atol=1e-12)
        np.testing.assert_allclose(front,x,atol=1e-12)

    def test_normalized_shape_is_invariant_to_sfm_scale_and_translation(self):
        points=np.random.default_rng(5).normal(size=(500,3))
        points[:,2]*=3
        a,_,height=normalized_coordinates(points,np.zeros(3),np.array([0,0,1]),np.array([1,0,0]))
        offset=np.array([4.,8.,-2.]);scale=17
        b,_,new_height=normalized_coordinates(points*scale+offset,offset,np.array([0,0,1]),np.array([1,0,0]))
        np.testing.assert_allclose(a,b,atol=1e-12)
        self.assertAlmostEqual(new_height,height*scale)

    def test_degenerate_camera_orbit_is_rejected(self):
        cameras=[dict(position=[i,0,0],rotation=np.eye(3)) for i in range(5)]
        with self.assertRaisesRegex(ValueError,'orbit plane'):
            fit_camera_ring(cameras)


if __name__=='__main__':
    unittest.main()
