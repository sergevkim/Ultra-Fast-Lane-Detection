import json
import os
from pathlib import Path

import bezier
import cv2
import numpy as np
import tqdm

from lane import (Lane, calc_max_distance_between_lanes, draw_middle_lane,
                  fill_spaces_between_lanes, get_horizontal_curves)


def convert(
    images_path,
    frames_info,
    culane_rgb_path,
    culane_gt_path,
    culane_list_path,
    limit: int = 30,
) -> None:
    if limit == -1:
        limit = len(frames_info)

    horizontal_curves = get_horizontal_curves()
    list_file = open(culane_list_path, 'w')

    for frame_index in tqdm.tqdm(range(min(limit, len(frames_info)))):
        frame_info = frames_info[frame_index]
        if 'labels' not in frame_info.keys():
            continue

        img = cv2.imread(str(images_path / frame_info['name']))
        gt = np.zeros_like(img)
        lanes = list()

        for i, lane in enumerate(frame_info['labels']):
            if lane['category'] in ['crosswalk', 'road curb']:
                continue

            lane_nodes = np.array(lane['poly2d'][0]['vertices'])
            lane_curve = bezier.Curve(
                lane_nodes.T,
                degree=lane_nodes.shape[0] - 1,
            )
            lane = Lane(
                index=i,
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

        forbidden_indices = set()

        for i in range(len(lanes)):
            for j in range(i + 1, len(lanes)):
                lane_1 = lanes[i]
                lane_2 = lanes[j]
                dist = calc_max_distance_between_lanes(lane_1, lane_2)
                if dist < 100:
                    #gt = draw_middle_lane(gt, lane_1, lane_2)
                    gt = fill_spaces_between_lanes(gt, lane_1, lane_2)
                    forbidden_indices.add(i)
                    forbidden_indices.add(j)

        for i, lane in enumerate(lanes):
            if i not in forbidden_indices:
                gt = lane.draw(gt)

        img = cv2.copyMakeBorder(
            img[65:-65],
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
        rgb_img_filename = '/rgb_images/' + frame_info['name']
        gt_img_filename = '/gt/' + frame_info['name']
        cv2.imwrite(rgb_img_filename, img)
        #gt = gt[:, :, 1]
        #print('gt', gt.shape)
        cv2.imwrite(gt_img_filename, gt)
        #from PIL import Image
        #print('!', len(Image.open(culane_gt_path / frame_info['name']).split()))
        list_file.write(f'{rgb_img_filename} {gt_img_filename} 1 1 1 1\n')

    list_file.close()


def main():
    bdd100k_as_culane_path = Path('/home/sergei/Downloads/bdd100k_as_culane')
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
        limit=-1,
    )
    convert(
        bdd100k_val_images_path,
        val_frames_info,
        bdd100k_as_culane_images_path,
        bdd100k_as_culane_gt_path,
        bdd100k_as_culane_val_list_path,
        limit=-1,
    )


if __name__ == "__main__":
    main()
