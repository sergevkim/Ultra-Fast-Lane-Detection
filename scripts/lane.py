import bezier
import cv2
import numpy as np


class Lane:
    def __init__(
        self,
        index,
        curve,
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

    def draw(self, image, color=None, radius=None):
        if color is None:
            color = self.default_color

        if radius is None:
            radius = self.point_radius

        prev_center = None

        for index in self.point_indices:
            center = (
                int(self.x_coords[index]),
                int(self.y_coords[index]),
            )

            image = cv2.circle(
                image,
                center,
                radius=radius,
                color=color,
                thickness=-1,
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


def calc_max_distance_between_lanes(
    lane_1: Lane,
    lane_2: Lane,
) -> float:
    max_distance = 0
    diff_counter = 0

    for index in lane_1.point_indices:
        if index not in lane_2.point_indices:
            diff_counter += 1
        else:
            x_1 = lane_1.x_coords[index]
            x_2 = lane_2.x_coords[index]
            cur_distance = abs(x_1 - x_2)
            max_distance = max(cur_distance, max_distance)

    for index in lane_2.point_indices:
        if index not in lane_1.point_indices:
            diff_counter += 1

    if diff_counter > 100:
        return 1000
    else:
        return max_distance


def fill_spaces_between_lanes(
    image,
    lane_1: Lane,
    lane_2: Lane,
):
    for index in lane_1.point_indices:
        if index in lane_2.point_indices:
            x_1 = lane_1.x_coords[index]
            x_2 = lane_2.x_coords[index]
            y_1 = lane_1.y_coords[index]
            y_2 = lane_2.y_coords[index]

            for bias_y in range(-1, 2):
                image = cv2.line(
                    image,
                    (x_1, y_1+bias_y),
                    (x_2, y_2),
                    color=(1, 1, 1),
                    thickness=1,
                )
                image = cv2.line(
                    image,
                    (x_1, y_1),
                    (x_2, y_2+bias_y),
                    color=(1, 1, 1),
                    thickness=1,
                )

    return image


def draw_middle_lane(
    image,
    lane_1: Lane,
    lane_2: Lane,
):
    prev_center = None

    for index in lane_1.point_indices:
        if index not in lane_2.point_indices:
            continue

        x_1 = lane_1.x_coords[index]
        x_2 = lane_2.x_coords[index]
        y_1 = lane_1.y_coords[index]
        y_2 = lane_2.y_coords[index]

        center = (
            int((x_1 + x_2) / 2),
            int((y_1 + y_2) / 2),
        )

        if prev_center is not None and abs(prev_center[1] - center[1]) < 5:
            image = cv2.line(
                image,
                prev_center,
                center,
                color=(1, 1, 1),
                thickness=15,
            )

        prev_center = center

    return image


def get_horizontal_curves():
    horizontal_curves = list()

    for i in range(0, 800, 1):
        nodes = np.asarray([[0, 1280], [i, i]])

        horizontal_curve = bezier.Curve(
            nodes,
            degree=1,
        )
        horizontal_curves.append(horizontal_curve)

    return horizontal_curves
