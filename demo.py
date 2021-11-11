import os

import cv2
import numpy as np
import scipy.special
import torch
import torchvision.transforms as transforms
import tqdm

from data.constant import culane_row_anchor, tusimple_row_anchor
from data.dataset import LaneTestDataset
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    dist_print('start testing...')
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    net = parsingNet(
        pretrained=False,
        backbone=cfg.backbone,
        cls_dim=(cfg.griding_num+1,cls_num_per_lane,4),
        use_aux=False
    ).cuda() # we dont need auxiliary segmentation in testing

    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    img_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    if cfg.dataset == 'CULane':
        splits = ['test_nodrops.txt']
        """
        splits = [
            'test0_normal.txt',
            'test1_crowd.txt',
            'test2_hlight.txt',
            'test3_shadow.txt',
            'test4_noline.txt',
            'test5_arrow.txt',
            'test6_curve.txt',
            'test7_cross.txt',
            'test8_night.txt',
        ]"""
        datasets = [
            LaneTestDataset(
                cfg.data_root,
                os.path.join(cfg.data_root, 'list/test_split/'+split),
                img_transform=img_transforms,
            )
            for split in splits
        ]
        img_w, img_h = 1640, 590
        row_anchor = culane_row_anchor
    elif cfg.dataset == 'Tusimple':
        splits = ['test.txt']
        datasets = [
            LaneTestDataset(
                cfg.data_root,
                os.path.join(cfg.data_root, split),
                img_transform=img_transforms,
            ) for split in splits
        ]
        img_w, img_h = 1280, 720
        row_anchor = tusimple_row_anchor
    else:
        raise NotImplementedError

    for split, dataset in zip(splits, datasets):
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1
        )
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        #video_file = split[:-3]+'avi'
        video_file = 'videos/culane_padding_v5_nodrops.mp4'
        #video_file = 'wayray_resized.avi'
        print(video_file)
        vout = cv2.VideoWriter(video_file, fourcc , 30.0, (img_w, img_h))

        for i, data in enumerate(tqdm.tqdm(loader)):
            imgs, names = data
            imgs = imgs.cuda()
            import time
            s = time.time()
            with torch.no_grad():
                out = net(imgs)
            #print('?????', time.time() - s)

            col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
            col_sample_w = col_sample[1] - col_sample[0]

            out_j = out[0].data.cpu().numpy()
            #print('outj0 shape: ', out_j.shape)
            out_j = out_j[:, ::-1, :]
            prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
            #print(type(prob))
            max_prob = np.max(prob, axis=0)
            idx = np.arange(cfg.griding_num) + 1
            idx = idx.reshape(-1, 1, 1)
            loc = np.sum(prob * idx, axis=0)
            out_j = np.argmax(out_j, axis=0)
            loc[out_j == cfg.griding_num] = 0
            out_j = loc
            #print('!!!!!', time.time() - s)
            #print('loc shape: ', loc.shape)
            #print('prob shape: ', prob[:,0,0])
            #print('!!!', imgs.shape)
            #print('col sample w: ', col_sample_w)

            # import pdb; pdb.set_trace()
            vis = cv2.imread(os.path.join(cfg.data_root,names[0]))
            vis = vis[290:-200,140:-140,:]
            #top, bottom, left, right = 0, 0, 180, 180
            #vis = cv2.copyMakeBorder(vis, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
            #vis = vis[130:,:,:]
            #print("vis shape: ", vis.shape, (1640, 590))
            #vis = cv2.resize(vis, (1640, 590))
            #vis = vis[250:, :, :]
            #vis = cv2.resize(vis, (1280, 720))
            #print('outj shape: ', out_j.shape)

            for i in range(out_j.shape[1]):
                wtf = np.sum(out_j[:, i] != 0)
                if wtf <= 2:
                    continue

                for k in range(out_j.shape[0]):
                    #print('wtf: ', wtf)
                    #print('KI: ', out_j[k, i])
                    if out_j[k, i] > 0:
                        ppp = (
                            int(out_j[k, i] * col_sample_w * img_w / 800) - 1,
                            int(img_h * (row_anchor[cls_num_per_lane - 1 - k] / 288)) - 1
                        )
                        ppp2 = (
                            ppp[0]+15,
                            ppp[1]+15,
                        )
                        cv2.circle(
                            vis,
                            ppp,
                            5,
                            (0,255,0),
                            -1,
                        )
                        if k % 2 == 0:
                            cv2.putText(
                                vis,
                                #'0.25',
                                f'{max_prob[k, i]:.2f}',
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.75,
                                org=ppp2,
                                color=(0, 0, 255),
                                thickness=2,
                            )
            vout.write(vis)

        vout.release()
        print('released!')
