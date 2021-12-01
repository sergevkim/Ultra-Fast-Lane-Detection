import typing as tp

import bezier
import cv2
import numpy as np


class Lane:
    def __init__(
        self,
        index,
        curve: bezier.Curve,
        horizontal_curves,
        point_radius: int = 1,
    ):
        self.index = index
        self.curve = curve
        self.n_points = 0
        self.point_indices = set()
        self.x_coords = dict()
        self.y_coords = dict()
        self.x_min = None
        self.x_max = None
        self.y_min = None
        self.y_max = None

        self.get_points_from_horizontal_curves_(horizontal_curves)

        self.default_color = (1, 1, 1)
        self.point_radius = point_radius

    def get_points_from_horizontal_curves_(self, horizontal_curves) -> None:
        for i, horizontal_curve in enumerate(horizontal_curves):
            intersections = self.curve.intersect(horizontal_curve)
            intersections = np.ascontiguousarray(intersections)
            intersection_point = self.curve.evaluate_multi(intersections[0, :])

            if intersection_point.shape[1] != 0:
                self.n_points += 1
                self.point_indices.add(i)
                x = int(intersection_point[0][0])
                y = int(intersection_point[1][0])
                self.x_coords[i] = x
                self.y_coords[i] = y

                if self.y_min is None:
                    self.x_min = x
                    self.x_max = x
                    self.y_min = y
                    self.y_max = y
                else:
                    self.x_min = min(x, self.x_min)
                    self.x_max = max(x, self.x_max)
                    self.y_min = min(y, self.y_min)
                    self.y_max = max(y, self.y_max)

    def draw(
        self,
        image,
        cluster_index: tp.Optional[int] = None,
    ) -> tp.Any:
        color = (cluster_index, cluster_index, cluster_index)
        prev_center = None

        for index in self.point_indices:
            center = (
                int(self.x_coords[index]),
                int(self.y_coords[index]),
            )

            if prev_center is not None:
                if abs(prev_center[1] - center[1]) < 5:
                    image = cv2.line(
                        image,
                        prev_center,
                        center,
                        color=color,
                        thickness=15,
                    )

            prev_center = center

        return image
