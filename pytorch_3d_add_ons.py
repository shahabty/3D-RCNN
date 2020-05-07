#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import math
import numpy as np
from typing import Tuple
import torch
import torch.nn.functional as F

from pytorch3d.transforms import Rotate, Transform3d, Translate

from pytorch3d.renderer.utils import TensorProperties, convert_to_tensors_and_broadcast

# Default values for rotation and translation matrices.
r = np.expand_dims(np.eye(3), axis=0)  # (1, 3, 3)
t = np.expand_dims(np.zeros(3), axis=0)  # (1, 3)


class OpenGLRealPerspectiveCameras(TensorProperties):
    """
    A class which stores a batch of parameters to generate a batch of
    projection matrices using the OpenGL convention for a perspective camera.

    The extrinsics of the camera (R and T matrices) can also be set in the
    initializer or passed in to `get_full_projection_transform` to get
    the full transformation from world -> screen.

    The `transform_points` method calculates the full world -> screen transform
    and then applies it to the input points.

    The transforms can also be returned separately as Transform3d objects.
    """

    def __init__(
        self,
        focal_length=1.0,
        principal_point=((0.0, 0.0),),
        R=r,
        T=t,
        znear=0.01,
        zfar=100.0,
        x0=0,
        y0=0,
        w=640,
        h=480,
        device="cpu",
    ):
        """
        __init__(self, znear, zfar, R, T, device) -> None  # noqa

        Args:
            znear: near clipping plane of the view frustrum.
            zfar: far clipping plane of the view frustrum.
            R: Rotation matrix of shape (N, 3, 3)
            T: Translation matrix of shape (N, 3)
            device: torch.device or string
        """
        # The initializer formats all inputs to torch tensors and broadcasts
        # all the inputs to have the same batch dimension where necessary.
        super().__init__(
            device=device,
            focal_length=focal_length,
            principal_point=principal_point,
            R=R,
            T=T,
            znear=znear,
            zfar=zfar,
            x0=x0,
            y0=y0,
            h=h,
            w=w,
        )

    def get_projection_transform(self, **kwargs) -> Transform3d:
        """
        Calculate the OpenGL perpective projection matrix with a symmetric
        viewing frustrum. Use column major order.

        Args:
            **kwargs: parameters for the projection can be passed in as keyword
                arguments to override the default values set in `__init__`.

        Return:
            P: a Transform3d object which represents a batch of projection
            matrices of shape (N, 3, 3)

        .. code-block:: python
            q = -(far + near)/(far - near)
            qn = -2*far*near/(far-near)

            P.T = [
                    [2*fx/w,     0,           0,  0],
                    [0,          -2*fy/h,     0,  0],
                    [(2*px-w)/w, (-2*py+h)/h, -q, 1],
                    [0,          0,           qn, 0],
                ]
                sometimes P[2,:] *= -1, P[1, :] *= -1
        """
        znear = kwargs.get("znear", self.znear)  # pyre-ignore[16]
        zfar = kwargs.get("zfar", self.zfar)  # pyre-ignore[16]
        x0 = kwargs.get("x0", self.x0)  # pyre-ignore[16]
        y0 = kwargs.get("y0", self.y0)  # pyre-ignore[16]
        w = kwargs.get("w", self.w)  # pyre-ignore[16]
        h = kwargs.get("h", self.h)  # pyre-ignore[16]
        principal_point = kwargs.get(
            "principal_point", self.principal_point
        )  # pyre-ignore[16]
        focal_length = kwargs.get(
            "focal_length", self.focal_length
        )  # pyre-ignore[16]

        if not torch.is_tensor(focal_length):
            focal_length = torch.tensor(focal_length, device=self.device)

        if len(focal_length.shape) in (0, 1) or focal_length.shape[1] == 1:
            fx = fy = focal_length
        else:
            fx, fy = focal_length.unbind(1)

        if not torch.is_tensor(principal_point):
            principal_point = torch.tensor(principal_point, device=self.device)
        px, py = principal_point.unbind(1)

        P = torch.zeros(
            (self._N, 4, 4), device=self.device, dtype=torch.float32
        )
        ones = torch.ones((self._N), dtype=torch.float32, device=self.device)

        # NOTE: In OpenGL the projection matrix changes the handedness of the
        # coordinate frame. i.e the NDC space postive z direction is the
        # camera space negative z direction. This is because the sign of the z
        # in the projection matrix is set to -1.0.
        # In pytorch3d we maintain a right handed coordinate system throughout
        # so the so the z sign is 1.0.
        z_sign = 1.0
        # define P.T directly
        P[:, 0, 0] = 2.0 * fx / w
        P[:, 1, 1] = -2.0 * fy / h
        P[:, 2, 0] = -(-2 * px + w + 2 * x0) / w
        P[:, 2, 1] = -(+2 * py - h + 2 * y0) / h
        P[:, 2, 3] = z_sign * ones

        # NOTE: This part of the matrix is for z renormalization in OpenGL
        # which maps the z to [-1, 1]. This won't work yet as the torch3d
        # rasterizer ignores faces which have z < 0.
        # P[:, 2, 2] = z_sign * (far + near) / (far - near)
        # P[:, 2, 3] = -2.0 * far * near / (far - near)
        # P[:, 2, 3] = z_sign * torch.ones((N))

        # NOTE: This maps the z coordinate from [0, 1] where z = 0 if the point
        # is at the near clipping plane and z = 1 when the point is at the far
        # clipping plane. This replaces the OpenGL z normalization to [-1, 1]
        # until rasterization is changed to clip at z = -1.
        P[:, 2, 2] = z_sign * zfar / (zfar - znear)
        P[:, 3, 2] = -(zfar * znear) / (zfar - znear)

        # OpenGL uses column vectors so need to transpose the projection matrix
        # as torch3d uses row vectors.
        transform = Transform3d(device=self.device)
        transform._matrix = P
        return transform

    def clone(self):
        other = OpenGLRealPerspectiveCameras(device=self.device)
        return super().clone(other)

    def get_camera_center(self, **kwargs):
        """
        Return the 3D location of the camera optical center
        in the world coordinates.

        Args:
            **kwargs: parameters for the camera extrinsics can be passed in
                as keyword arguments to override the default values
                set in __init__.

        Setting T here will update the values set in init as this
        value may be needed later on in the rendering pipeline e.g. for
        lighting calculations.

        Returns:
            C: a batch of 3D locations of shape (N, 3) denoting
            the locations of the center of each camera in the batch.
        """
        w2v_trans = self.get_world_to_view_transform(**kwargs)
        P = w2v_trans.inverse().get_matrix()
        # the camera center is the translation component (the first 3 elements
        # of the last row) of the inverted world-to-view
        # transform (4x4 RT matrix)
        C = P[:, 3, :3]
        return C

    def get_world_to_view_transform(self, **kwargs) -> Transform3d:
        """
        Return the world-to-view transform.

        Args:
            **kwargs: parameters for the camera extrinsics can be passed in
                as keyword arguments to override the default values
                set in __init__.

        Setting R and T here will update the values set in init as these
        values may be needed later on in the rendering pipeline e.g. for
        lighting calculations.

        Returns:
            T: a Transform3d object which represents a batch of transforms
            of shape (N, 3, 3)
        """
        R = self.R = kwargs.get("R", self.R)  # pyre-ignore[16]
        T = self.T = kwargs.get("T", self.T)  # pyre-ignore[16]
        if T.shape[0] != R.shape[0]:
            msg = "Expected R, T to have the same batch dimension; got %r, %r"
            raise ValueError(msg % (R.shape[0], T.shape[0]))
        if T.dim() != 2 or T.shape[1:] != (3,):
            msg = "Expected T to have shape (N, 3); got %r"
            raise ValueError(msg % repr(T.shape))
        if R.dim() != 3 or R.shape[1:] != (3, 3):
            msg = "Expected R to have shape (N, 3, 3); got %r"
            raise ValueError(msg % R.shape)

        # Create a Transform3d object
        T = Translate(T, device=T.device)
        R = Rotate(R, device=R.device)
        world_to_view_transform = R.compose(T)
        return world_to_view_transform

    def get_full_projection_transform(self, **kwargs) -> Transform3d:
        """
        Return the full world-to-screen transform composing the
        world-to-view and view-to-screen transforms.

        Args:
            **kwargs: parameters for the projection transforms can be passed in
                as keyword arguments to override the default values
                set in __init__.

        Setting R and T here will update the values set in init as these
        values may be needed later on in the rendering pipeline e.g. for
        lighting calculations.

        Returns:
            T: a Transform3d object which represents a batch of transforms
            of shape (N, 3, 3)
        """
        self.R = kwargs.get("R", self.R)  # pyre-ignore[16]
        self.T = kwargs.get("T", self.T)  # pyre-ignore[16]

        world_to_view_transform = self.get_world_to_view_transform(
            R=self.R, T=self.T
        )
        view_to_screen_transform = self.get_projection_transform(**kwargs)
        return world_to_view_transform.compose(view_to_screen_transform)

    def transform_points(self, points, **kwargs) -> torch.Tensor:
        """
        Transform input points from world to screen space.

        Args:
            points: torch tensor of shape (..., 3).

        Returns
            new_points: transformed points with the same shape as the input.
        """
        world_to_screen_transform = self.get_full_projection_transform(**kwargs)
        return world_to_screen_transform.transform_points(points)
