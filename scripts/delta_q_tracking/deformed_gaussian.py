from __future__ import annotations


class DeformedGaussian:
    """Thin renderer-compatible wrapper that can override Gaussian xyz/rotation."""

    def __init__(self, base, deformed_xyz, deformed_rotation=None):
        self.base = base
        self.deformed_xyz = deformed_xyz
        self.deformed_rotation = deformed_rotation
        self.active_sh_degree = base.active_sh_degree
        self.max_sh_degree = base.max_sh_degree

    @property
    def get_xyz(self):
        return self.deformed_xyz

    @property
    def get_scaling(self):
        return self.base.get_scaling

    @property
    def get_rotation(self):
        return self.deformed_rotation if self.deformed_rotation is not None else self.base.get_rotation

    @property
    def get_features(self):
        return self.base.get_features

    @property
    def get_features_dc(self):
        return self.base.get_features_dc

    @property
    def get_features_rest(self):
        return self.base.get_features_rest

    @property
    def get_opacity(self):
        return self.base.get_opacity

    def get_covariance(self, scaling_modifier=1):
        return self.base.get_covariance(scaling_modifier)

    def get_exposure_from_name(self, image_name):
        return self.base.get_exposure_from_name(image_name)
