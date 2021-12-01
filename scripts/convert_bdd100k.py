import json
import os
import typing as tp
from pathlib import Path

import bezier
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from lane.lane import Lane
from lane.utils import (
    get_horizontal_curves,
    cluster_lanes,
    draw_lane_clusters,
)


def get_lanes(frame_info, horizontal_curves) -> tp.List[Lane]:
    lanes = list()

    for lane_index, lane_info in enumerate(frame_info['labels']):
        if lane_info['category'] in ['crosswalk', 'road curb']:
            continue

        lane_nodes = np.array(lane_info['poly2d'][0]['vertices'])
        lane_curve = bezier.Curve(
            lane_nodes.T,
            degree=lane_nodes.shape[0] - 1,
        )
        lane = Lane(
            index=lane_index,
            curve=lane_curve,
            horizontal_curves=horizontal_curves,
        )

        if lane.y_min is None:
            continue
        else:
            delta_y = lane.y_max - lane.y_min

        if delta_y > 50:
            delta_x = lane.x_max - lane.x_min
            if delta_x != 0 and delta_y / delta_x > 0.2:
                lanes.append(lane)

    return lanes


def process_frame(images_path, frame_info, horizontal_curves, mode):
    image = cv2.imread(str(images_path / frame_info['name']))
    if mode == 'write':
        gt = np.zeros_like(image)
    elif mode == 'show':
        gt = image.copy()

    lanes = get_lanes(frame_info, horizontal_curves=horizontal_curves)
    lane_clusters = cluster_lanes(lanes)
    gt = draw_lane_clusters(image=gt, lane_clusters=lane_clusters)

    image = cv2.copyMakeBorder(
        image[65:-65],
        top=0,
        bottom=0,
        left=180,
        right=180,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )
    gt = cv2.copyMakeBorder(
        gt[65:-65],
        top=0,
        bottom=0,
        left=180,
        right=180,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )

    return image, gt


def convert(
    images_path,
    frames_info,
    culane_rgb_path = None,
    culane_gt_path = None,
    culane_list_path = None,
    limit: int = 30,
    mode: str = 'write',
) -> None:
    if limit == -1:
        limit = len(frames_info)

    horizontal_curves = get_horizontal_curves()

    list_file = open(culane_list_path, 'w')

    for frame_index in tqdm.tqdm(range(min(limit, len(frames_info)))):
        frame_info = frames_info[frame_index]
        if 'labels' not in frame_info.keys():
            continue

        image, gt = process_frame(
            images_path=images_path,
            frame_info=frame_info,
            horizontal_curves=horizontal_curves,
            mode=mode,
        )

        if mode == 'write':
            assert culane_gt_path is not None, 'Some paths are None'
            assert culane_rgb_path is not None, 'Some paths are None'
            assert culane_list_path is not None, 'Some paths are None'

            rgb_image_filename = '/rgb_images/' + frame_info['name'].replace('jpg', 'png')
            gt_image_filename = '/gt/' + frame_info['name'].replace('jpg', 'png')

            list_line = f'{rgb_image_filename} {gt_image_filename} 1 1 1 1\n'
            list_file.write(list_line)

            cv2.imwrite(str(culane_rgb_path / frame_info['name'].replace('jpg', 'png')), image)
            cv2.imwrite(str(culane_gt_path / frame_info['name'].replace('jpg', 'png')), gt)
        elif mode == 'show':
            plt.imshow(gt)
            plt.show()

    list_file.close()


def main():
    bdd100k_as_culane_path = Path('/home/sergei/Downloads/bdd100k_as_culane_small')
    bdd100k_as_culane_images_path = bdd100k_as_culane_path / 'rgb_images'
    bdd100k_as_culane_gt_path = bdd100k_as_culane_path / 'gt'
    bdd100k_as_culane_train_list_path = \
        bdd100k_as_culane_path / 'list/train_gt.txt'
    bdd100k_as_culane_val_list_path = bdd100k_as_culane_path / 'list/val_gt.txt'

    if not os.path.exists(bdd100k_as_culane_path / 'list'):
        os.makedirs(bdd100k_as_culane_path / 'list')
    if not os.path.exists(bdd100k_as_culane_images_path):
        os.makedirs(bdd100k_as_culane_images_path)
    if not os.path.exists(bdd100k_as_culane_gt_path):
        os.makedirs(bdd100k_as_culane_gt_path)

    bdd100k_path = Path('/home/sergei/Downloads/bdd100k')
    bdd100k_images_path = bdd100k_path / 'images/100k'
    bdd100k_train_images_path = bdd100k_images_path / 'train'
    bdd100k_val_images_path = bdd100k_images_path / 'val'
    bdd100k_labels_path = bdd100k_path / 'labels'
    bdd100k_lane_labels_path = bdd100k_labels_path / 'lane/polygons'
    bdd100k_train_lane_labels_path = \
        bdd100k_lane_labels_path / 'lane_train.json'
    bdd100k_val_lane_labels_path = bdd100k_lane_labels_path / 'lane_val.json'

    train_frames_info = dict()
    with open(bdd100k_train_lane_labels_path) as f:
        train_frames_info = json.load(f)

    val_frames_info = dict()
    with open(bdd100k_val_lane_labels_path) as f:
        val_frames_info = json.load(f)

    convert(
        bdd100k_train_images_path,
        train_frames_info,
        bdd100k_as_culane_images_path,
        bdd100k_as_culane_gt_path,
        bdd100k_as_culane_train_list_path,
        limit=2,
    )
    convert(
        bdd100k_val_images_path,
        val_frames_info,
        bdd100k_as_culane_images_path,
        bdd100k_as_culane_gt_path,
        bdd100k_as_culane_val_list_path,
        limit=2,
    )


if __name__ == "__main__":
    main()
