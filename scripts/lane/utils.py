import typing as tp

import bezier
import cv2
import numpy as np

from .lane import Lane


def get_horizontal_curves() -> tp.List[bezier.Curve]:
    horizontal_curves = list()

    for i in range(0, 800, 1):
        nodes = np.asarray([[0, 1280], [i, i]])

        horizontal_curve = bezier.Curve(
            nodes,
            degree=1,
        )
        horizontal_curves.append(horizontal_curve)

    return horizontal_curves


def cluster_lanes(lanes: tp.List[Lane]) -> tp.List[tp.List[Lane]]:
    lane_clusters = list()
    index2cluster = dict()

    for lane_index, current_lane in enumerate(lanes):
        new_cluster_flag = True

        for prev_lane_index in range(lane_index):
            dist = calc_max_distance_between_lanes(
                lanes[prev_lane_index],
                current_lane,
            )
            if dist < 100:
                cluster_for_current_lane = index2cluster[prev_lane_index]
                lane_clusters[cluster_for_current_lane].append(current_lane)
                new_cluster_flag = False
                break

        if new_cluster_flag:
            lane_clusters.append([current_lane])
            index2cluster[lane_index] = len(lane_clusters) - 1

    return lane_clusters


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


def fill_spaces_between_two_lanes(
    image,
    lane_1: Lane,
    lane_2: Lane,
    cluster_index: int,
) -> tp.Any:
    color = (cluster_index, cluster_index, cluster_index)

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
                    color=color,
                    thickness=1,
                )
                image = cv2.line(
                    image,
                    (x_1, y_1),
                    (x_2, y_2+bias_y),
                    color=color,
                    thickness=1,
                )

    return image


def fill_spaces_between_all_lanes(
    image,
    lanes: tp.List[Lane],
    cluster_index: int,
) -> tp.Any:
    for index_1 in range(len(lanes)):
        for index_2 in range(index_1 + 1, len(lanes)):
            lane_1 = lanes[index_1]
            lane_2 = lanes[index_2]
            image = fill_spaces_between_two_lanes(
                image=image,
                lane_1=lane_1,
                lane_2=lane_2,
                cluster_index=cluster_index,
            )

    return image

''''
def draw_middle_lane(
    image,
    lane_1: Lane,
    lane_2: Lane,
) -> tp.Any:
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
'''

def draw_lane_clusters(image, lane_clusters: tp.List[tp.List[Lane]]) -> tp.Any:
    for cluster_index, lane_cluster in enumerate(lane_clusters):
        for lane in lane_cluster:
            image = lane.draw(image, cluster_index=cluster_index)
            print(cluster_index, len(lane_clusters))

            #import ipdb; ipdb.set_trace()
        print('------------')

        #if len(lane_cluster) > 0:
        #    image = fill_spaces_between_all_lanes(
        #        image=image,
        #        lanes=lane_cluster,
        #        cluster_index=(cluster_index+1)*10,
        #    )

    return image
