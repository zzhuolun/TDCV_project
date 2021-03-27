import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from unet import UNet
from utils import NOCSDataset, NOCSTestset, id_local_to_global, SingleData, SingleTestdata, mksquare_reverse, mspd, \
    _make_square, align_pred_gt
import pickle
sys.path.append('model_projector/')
import glob
from PIL import Image
import random

K = np.array([591.0125, 0.0, 322.525, 0.0, 590.16775, 244.11084, 0.0, 0.0, 1.0])
K = K.reshape((3, 3))
resize_input = 64


def predict(model_path, test_dir, cls=[1, 2, 3, 4, 5, 6], gpu=0, test_number=10):
    """
    Show the effects of xyzmask prediction and pose estimation for test images (i.e. real test)
    Args:
        model_path: directory which stores the models
        test_dir: director to testset
        cls: which classes to be evaluted (e.g. 1/[1,2,3,4,5,5])
        gpu: gpu id to run
        test_number: the number of images to be visualized
    """
    if isinstance(cls, int):
        cls = (cls,)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print("device: ", f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    nets = {}
    for i in cls:
        correspondence_block = UNet()
        correspondence_block.load_state_dict(
            torch.load(os.path.join(model_path, f'real_synthetic_sym{i}.pt'), map_location=device))
        correspondence_block = correspondence_block.to(device)
        correspondence_block.eval()
        nets[i] = correspondence_block
    test_data = NOCSTestset(test_dir, 'test_imgs.txt', cls,
                            transform=transforms.Compose([transforms.Resize(resize_input)]))
    num_workers = 0
    test_indices = list(range(len(test_data)))
    np.random.shuffle(test_indices)
    test_indices = test_indices[:test_number]
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1,
                                              sampler=SubsetRandomSampler(test_indices), num_workers=num_workers)

    for rgb, adr_rgb in test_loader:
        try:
            adr_rgb = adr_rgb[0]
            single_data = SingleTestdata(adr_rgb)
            cls_id = single_data.cls_id
            if cls_id not in cls:
                continue
            net = nets[cls_id]

            rgb = rgb.to(device)
            xmask_pred, ymask_pred, zmask_pred = net(rgb)
            xmask_pred = torch.argmax(xmask_pred[0], 0).to('cpu').detach().numpy()
            ymask_pred = torch.argmax(ymask_pred[0], 0).to('cpu').detach().numpy()
            zmask_pred = torch.argmax(zmask_pred[0], 0).to('cpu').detach().numpy()

            xyzmask_pred = np.dstack([xmask_pred, ymask_pred, zmask_pred]) / 255
            xyzmask_pred = np.clip(xyzmask_pred, 0, 1)

            rgb_uncropped = cv2.imread(single_data.adr_uncropped)  # the original uncropped image
            rgb_uncropped = cv2.cvtColor(rgb_uncropped, cv2.COLOR_BGR2RGB)

            # PNP Ransac using predicted xyzmask
            R, t = ransac_pnp(xyzmask_pred, K,
                              single_data, resize_input, rgb_uncropped)
            R_gt = single_data.R
            t_gt = single_data.t

            mspd_error, mspd_error_idx = mspd(R, t, R_gt, t_gt, K, single_data.vertices, single_data.get_sym())
            print(f"test image {adr_rgb}")
            print('class: ', cls_id)
            print('MSPD Error: ', mspd_error)
            print(mspd_error_idx)
            pose = Rt_to_H(R, t)
            pose_gt = Rt_to_H(R_gt, t_gt)
            # draw the bbx with poses
            img_bbx = create_bounding_box(rgb_uncropped, pose, pose_gt, single_data, K)
            rgb = rgb[0].to('cpu').detach().permute(1, 2, 0).numpy()
            xyzmask = cv2.imread(single_data.adr_coord)  # the original uncropped image
            xyzmask = cv2.cvtColor(xyzmask, cv2.COLOR_BGR2RGB)
            xyzmask = xyzmask[single_data.crop_bbx()[0]:single_data.crop_bbx()[2],
                      single_data.crop_bbx()[1]:single_data.crop_bbx()[3], :]
            fig, axs = plt.subplots(2, 2, figsize=(8, 8))
            axs[0, 0].imshow(rgb)
            axs[0, 0].set_title('rgb')
            axs[0, 1].imshow(xyzmask_pred)
            axs[0, 1].set_title('xyzmask_pred')
            axs[1, 0].imshow(xyzmask)
            axs[1, 0].set_title('xyzmask')
            fig.delaxes(axs[1, 1])
            plt.figure()
            plt.imshow(img_bbx)
            plt.show()
        except Exception as e:
            print(e)
            continue


def predict_train(test_json, cls, model_path, gpu=0, test_number=10):
    """
    Show the effects of xyzmask prediction and pose estimation for a single class of train images (i.e. real train and synthesized data)
    Args:
        test_json: json file which stores the img dir
        cls: the class of the imgs to be evalated
        model_path: the path to the model
        gpu: gpu_id to use
        test_number: the number of images to be visualized
    """
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print("device: ", f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    correspondence_block = UNet()
    correspondence_block.load_state_dict(torch.load(model_path, map_location=device))
    correspondence_block = correspondence_block.to(device)
    correspondence_block.eval()
    test_data = NOCSDataset(test_json, cls, transform=transforms.Compose([transforms.Resize(resize_input)]))

    num_workers = 0
    test_indices = list(range(len(test_data)))
    np.random.shuffle(test_indices)
    if test_number is not None:
        test_indices = test_indices[:test_number]
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1,
                                              sampler=SubsetRandomSampler(test_indices), num_workers=num_workers)
    for idx, (rgb, xmask, ymask, zmask, adr_rgb) in enumerate(test_loader):
        try:
            print(f"test image {idx} ({adr_rgb})")
            single_data = SingleData(adr_rgb[0])
            rgb = rgb.to(device)
            xmask = xmask.to(device)
            ymask = ymask.to(device)
            zmask = zmask.to(device)

            xmask_pred, ymask_pred, zmask_pred = correspondence_block(rgb)

            xmask_pred = torch.argmax(xmask_pred[0], 0).to('cpu').detach().numpy()
            ymask_pred = torch.argmax(ymask_pred[0], 0).to('cpu').detach().numpy()
            zmask_pred = torch.argmax(zmask_pred[0], 0).to('cpu').detach().numpy()

            xmask = xmask[0].to('cpu').detach().numpy()
            ymask = ymask[0].to('cpu').detach().numpy()
            zmask = zmask[0].to('cpu').detach().numpy()

            xyzmask = np.dstack([xmask, ymask, zmask]) / 255

            xyzmask_pred = np.dstack([xmask_pred, ymask_pred, zmask_pred]) / 255
            xyzmask_pred = np.clip(xyzmask_pred, 0, 1)

            rgb = rgb[0].to('cpu').detach().permute(1, 2, 0).numpy()

            rgb_uncropped = cv2.imread(single_data.adr_uncropped)  # the original uncropped image
            rgb_uncropped = cv2.cvtColor(rgb_uncropped, cv2.COLOR_BGR2RGB)

            #         estimate pose
            R, t = ransac_pnp(xyzmask_pred, single_data.intrinsics(),
                              single_data, resize_input, rgb_uncropped)
            R_gt, t_gt = single_data.gt_pose()
            mspd_error, mspd_error_idx = mspd(R, t, R_gt, t_gt, single_data.intrinsics(), single_data.get_model(),
                                              single_data.get_sym())

            print('MSPD Error: ', mspd_error)
            print(mspd_error_idx)
            pose = Rt_to_H(R, t)
            pose_gt = Rt_to_H(R_gt, t_gt)
            # draw the bbx with poses
            img_bbx = create_bounding_box(rgb_uncropped, pose, pose_gt, single_data, single_data.intrinsics())

            fig, axs = plt.subplots(2, 2)
            axs[0, 0].imshow(rgb)
            axs[0, 0].set_title('rgb')
            axs[0, 1].imshow(xyzmask)
            axs[0, 1].set_title('xyzmask')
            axs[1, 0].imshow(xyzmask_pred)
            axs[1, 0].set_title('xyzmask_pred')
            fig.delaxes(axs[1, 1])

            plt.figure()
            plt.imshow(img_bbx)
            plt.show()
        except Exception as e:
            print(e)
            continue


def test_mspd(cls, model_path, test_dir, batch_size=64, gpu=0, iter_print=10):
    """
    Compute MSPD error for a single class of testset
    Args:
        cls: the class to be evaluated
        model_path: path to the model
        test_dir: path to test dataset
        batch_size: batch size used in inference
        gpu: gpu id to use
        iter_print: print process in evaluating per iter_print batches

    Returns:
        list of mspd error for the class of test images
    """
    assert isinstance(cls, int)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print("device: ", f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    correspondence_block = UNet()
    correspondence_block.load_state_dict(torch.load(model_path))
    correspondence_block = correspondence_block.to(device)
    correspondence_block.eval()
    net = correspondence_block
    test_data = NOCSTestset(test_dir, 'test_imgs.txt', cls,
                            transform=transforms.Compose([transforms.Resize(resize_input)]))
    test_indices = list(range(len(test_data)))
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                              sampler=SubsetRandomSampler(test_indices))
    test_size = len(test_data)
    MSPD = []
    for idx, (rgb, adr_rgb) in enumerate(test_loader):
        try:
            t0 = time.time()
            rgb = rgb.to(device)
            xmask_pred, ymask_pred, zmask_pred = net(rgb)
            xmask_pred = torch.argmax(xmask_pred, 1).to('cpu').detach().numpy()
            ymask_pred = torch.argmax(ymask_pred, 1).to('cpu').detach().numpy()
            zmask_pred = torch.argmax(zmask_pred, 1).to('cpu').detach().numpy()
            xyzmask_pred = np.stack([xmask_pred, ymask_pred, zmask_pred], axis=-1) / 255
            xyzmask_pred = np.clip(xyzmask_pred, 0, 1)
            for batch_i in range(xyzmask_pred.shape[0]):
                single_data = SingleTestdata(adr_rgb[batch_i])
                rgb_uncropped = cv2.imread(single_data.adr_uncropped)  # the original uncropped image
                rgb_uncropped = cv2.cvtColor(rgb_uncropped, cv2.COLOR_BGR2RGB)

                R, t = ransac_pnp(xyzmask_pred[batch_i], K,
                                  single_data, resize_input, rgb_uncropped)
                R_gt = single_data.R
                t_gt = single_data.t

                mspd_error, mspd_error_idx = mspd(R, t, R_gt, t_gt, K, single_data.vertices, single_data.get_sym())
                MSPD.append(mspd_error)
            t_end = time.time()
            print(f'{(t_end - t0):.1f} seconds')
            if idx % iter_print == 0:
                print(f'{(idx + 1) * batch_size}/{test_size}', f'{(100 * batch_size * (idx + 1) / test_size):.0f}%')
        except Exception as e:
            if 'batch_i' in locals():
                print(adr_rgb[batch_i])
            print(e)
            continue

    return MSPD


def nocs_mspd(cls, noc_pred_dir, test_dir):
    """
    Compute MSPD error for NOCS implementation, given already generated pose of NOCS
    Args:
        cls: the class to be evaluated
        noc_pred_dir: the directory of the generated pose by NOCS
        test_dir: directory of test dataset

    Returns:
        list of mspd error for the class of test images
    """
    assert isinstance(cls, int)
    with open('test_imgs.txt', 'r') as f:
        test_imgs = json.load(f)
    test_ls = test_imgs[str(cls)]
    MSPD = []
    for pt in test_ls:
        try:
            # print(pt)
            single_data = SingleTestdata(os.path.join(test_dir, pt))
            R_gt = single_data.R
            t_gt = single_data.t

            pred_path = os.path.join(noc_pred_dir,
                                     f'results_test_scene_{single_data.scene_idx}_{single_data.img_idx:04d}.pkl')

            with open(pred_path, 'rb') as f:
                pred_data = pickle.load(f)

            aligned_id = align_pred_gt(pred_data, single_data.inst_id)
            if not aligned_id:
                continue
            H_pred = pred_data['pred_RTs'][aligned_id]
            R = H_pred[:3, :3]
            t = H_pred[:3, 3].reshape(3, 1)
            mspd_error, mspd_error_idx = mspd(R, t, R_gt, t_gt, K, single_data.vertices, single_data.get_sym())
            # print(mspd_error)
            MSPD.append(mspd_error)


        except Exception as e:
            print(e)
            print(pt)
            continue
    return MSPD


def ransac_pnp(xyzmask, intrinsics, single_data, resize_input, rgb, reproj_error=1.0):
    """
    PnPRansac based on 2d-3d correspondences
    Args:
        xyzmask: predicted xyzmask, encoding the 2d-3d correspondences
        intrinsics: camera intrinsic matrix
        single_data: an object of the SingleData class
        resize_input: value of resize during training dataloader
        rgb: original uncropped rgb image
        reproj_error: reproject error used in cv2.PnPRansac

    Returns:
        estimated rotation and translation
    """
    # only use the non-zero (non background) pixel for correspondences
    temp = xyzmask[:, :, 0] + xyzmask[:, :, 1] + xyzmask[:, :, 2]
    coord = temp.nonzero()
    image_coord = np.vstack(coord).T
    # 3d correspondences
    world_coord = xyzmask[image_coord[:, 0], image_coord[:, 1], :]
    world_coord = remap_bbx(world_coord, single_data)
    # 2d correspondences
    image_coord = coord_trans(image_coord, single_data, resize_input)
    image_coord = image_coord[:, ::-1]

    try:
        reval, R, t, inliers = cv2.solvePnPRansac(world_coord, image_coord.astype('float'),
                                                  intrinsics, distCoeffs=None, reprojectionError=reproj_error)
    except Exception as e:
        # here is for a bug in cv2.solvePnPRansac: even if more than 6 points provided, opencv sometimes still report the error:
        # pnpransac DLT algorithm needs at least 6 points for pose estimation from 3D-2D point correspondences.
        print('remove last element of correspondences')
        world_coord = world_coord[:-1, :]
        image_coord = image_coord[:-1, :]
        reval, R, t, inliers = cv2.solvePnPRansac(world_coord, image_coord.astype('float'),
                                                  intrinsics, distCoeffs=None, reprojectionError=reproj_error)
    fail_cnt = 0
    while not reval:
        # increase reproj_error if cv2.solvePnPRansac failed
        reproj_error *= 1.5
        reval, R, t, inliers = cv2.solvePnPRansac(world_coord, image_coord.astype('float'),
                                                  intrinsics, distCoeffs=None, reprojectionError=reproj_error)
        fail_cnt += 1
        if fail_cnt >= 12:
            break
    #         print('increase reprojection error to ', reproj_error)
    valid_coord = image_coord[inliers.flatten()]

    for points in valid_coord:
        cv2.circle(rgb, (points[0], points[1]), 4, (255, 0, 0), 1)
        # print(image_coord.shape)
        # print('number of inlier pixels: ', inliers.shape[0])
    return cv2.Rodrigues(R)[0], t


def Rt_to_H(R, t, inverse=False):
    H = np.zeros((4, 4))
    H[3, 3] = 1
    if inverse:
        t = - R.T @ t
        R = R.T
    H[:3, :3] = R
    H[:3, 3] = t.flatten()
    return H


def coord_trans(image_coord, single_data, resize_input):
    """
    transfer the local image plane coordinates in the cropped image to the coordinates of the whole scene
    Args:
        image_coord: local image coordinates before transfer
        single_data: an object of the SingleData class
        resize_input: value of resize during training dataloader

    Returns:
        the transferred image coordinates in the whole scene
    """
    image_coord = image_coord.astype('float')
    minx, miny, maxx, maxy = single_data.crop_bbx()
    before_resize = max(maxx - minx, maxy - miny)
    image_coord *= before_resize / resize_input
    offset_x, offset_y = mksquare_reverse(maxx - minx, maxy - miny)
    image_coord -= np.array([offset_x, offset_y])
    image_coord += np.array([minx, miny])
    image_coord = image_coord.astype('int')
    #     if (image_coord>=0).all():
    #         for points in image_coord:
    #             cv2.circle(rgb, (points[1],points[0]), 4, (255, 0, 0), 2)
    return image_coord


def remap_bbx(world_coord, single_data):
    """
    remap the normalized xyzmask (value from 0-1) to 3d point coordinates
    Args:
        world_coord: the 3d point coordinates extracted from the xyzmask before transfer, from 0-1
        single_data: an object of the SingleData class

    Returns:
        the transferred 3d point coordinates
    """
    corners_3D, _ = single_data.bbx_3d()
    minx, miny, minz = corners_3D[0]
    maxx, maxy, maxz = corners_3D[6]
    world_coord = world_coord * np.array([maxx - minx, maxy - miny, maxz - minz])
    world_coord = world_coord + np.array([minx, miny, minz])
    return world_coord


def create_bounding_box(img, pose_estimate, pose_gt, single_data, intrinsic_matrix):
    """
        Create a bounding box of both gt and estimated pose around the object
    Args:
        img: rgb image
        pose_estimate: estimated pose0
        pose_gt: ground truth pose
        single_data: an object of the SingleData class
        intrinsic_matrix: camera intrinsics

    Returns:
        the rgb image with bbx
    """
    corners_3D, draw_bbx = single_data.bbx_3d()
    # convert these 8 3D corners to 2D points
    ones = np.ones((corners_3D.shape[0], 1))
    homogenous_coordinate = np.append(corners_3D, ones, axis=1)
    # Perspective Projection to obtain 2D coordinates for masks
    color = [(0, 255, 0), (0, 0, 255)]
    for idx, pose in enumerate([pose_gt, pose_estimate]):

        homogenous_2D = intrinsic_matrix @ (pose[:3, :] @ homogenous_coordinate.T)

        coord_2D = homogenous_2D[:2, :] / homogenous_2D[2, :]
        coord_2D = ((np.floor(coord_2D)).T).astype(int)

        # Draw lines between these 8 points
        for vertex in draw_bbx:
            img = cv2.line(img, tuple(coord_2D[vertex[0]]), tuple(coord_2D[vertex[1]]), color[idx], 3)
    return img


def mspd_process(mspd, thres=1000):
    """
    process the raw mspd list values
    """
    mspd = np.array(mspd)
    outlier = np.sum(mspd > thres) / len(mspd)
    mspd_mean = np.mean(mspd[mspd <= thres])
    mspd_median = np.median(mspd[mspd <= thres])
    return mspd_mean, outlier, mspd_median


def mspd_log(model_basename, gpu):
    """
    log the mean, median, etc of mspd to a dictionary for our method, excluding some outliers
    """
    mspd_dict = defaultdict(dict)
    for i in range(1, 7):
        print('class ', i)
        mspd_error = test_mspd(i, f'ckpt/{model_basename}{i}.pt',
                               '/home/zhuolun/evalmia/arxiv/__pycache__/data/real/test', gpu=gpu)
        mspd_mean, outlier, mspd_median = mspd_process(mspd_error)
        mspd_dict[i]['mspd'] = mspd_error
        mspd_dict[i]['mean'] = mspd_mean
        mspd_dict[i]['outlier'] = outlier
        mspd_dict[i]['median'] = mspd_median
        print('mean: ', mspd_mean)
        print('outlier percentage: ', outlier)
        print('median', mspd_median)
    return mspd_dict


def mspd_log_nocs(model_dir, test_dir):
    """
    log the mean, median, etc of mspd to a dictionary for NOCS implementation
    """
    mspd_dict = defaultdict(dict)
    for i in range(1, 7):
        print('class ', i)
        mspd_error = nocs_mspd(i, model_dir, test_dir)
        mspd_mean, outlier, mspd_median = mspd_process(mspd_error)
        mspd_dict[i]['mspd'] = mspd_error
        mspd_dict[i]['mean'] = mspd_mean
        mspd_dict[i]['outlier'] = outlier
        mspd_dict[i]['median'] = mspd_median
        print('mean: ', mspd_mean)
        print('outlier percentage: ', outlier)
        print('median', mspd_median)
    return mspd_dict


def draw_bbx_test(test_dir, model_name, gpu=3):
    """
    draw the bbx for whole scenes of the testdataset
    """
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print("device: ", f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    img_ls = glob.glob(os.path.join(test_dir, 'scene_*/*_color.png'))
    random.shuffle(img_ls)
    transform = transforms.Resize(resize_input)
    nets = {}
    for i in range(1, 7):
        correspondence_block = UNet()
        correspondence_block.load_state_dict(
            torch.load(f'ckpt/{model_name}{i}.pt', map_location=device))
        correspondence_block = correspondence_block.to(device)
        correspondence_block.eval()
        nets[i] = correspondence_block
    for idx, img_path in enumerate(img_ls):
        scene_id = img_path.split('/')[-2][-1]
        img_id = img_path.split('/')[-1][:4]
        obj_ls = glob.glob(str(Path(img_path).parents[0] / 'cropped' / img_id) + '*')
        rgb_uncropped = cv2.imread(img_path)  # the original uncropped image
        rgb_uncropped = cv2.cvtColor(rgb_uncropped, cv2.COLOR_BGR2RGB)
        # rgb_uncropped_mask = Image.fromarray(rgb_uncropped.astype(np.uint8))
        for adr_rgb in obj_ls:
            single_data = SingleTestdata(adr_rgb)
            cls_id = single_data.cls_id
            net = nets[cls_id]

            rgb = Image.open(adr_rgb)
            to_tensor = transforms.ToTensor()
            rgb = _make_square(rgb)
            rgb = transform(rgb)
            rgb = to_tensor(rgb)
            rgb = torch.unsqueeze(rgb, 0)
            rgb = rgb.to(device)
            xmask_pred, ymask_pred, zmask_pred = net(rgb)
            xmask_pred = torch.argmax(xmask_pred[0], 0).to('cpu').detach().numpy()
            ymask_pred = torch.argmax(ymask_pred[0], 0).to('cpu').detach().numpy()
            zmask_pred = torch.argmax(zmask_pred[0], 0).to('cpu').detach().numpy()
            xyzmask_pred = np.stack([xmask_pred, ymask_pred, zmask_pred], axis=-1) / 255
            xyzmask_pred = np.clip(xyzmask_pred, 0, 1)

            R, t = ransac_pnp(xyzmask_pred, K,
                              single_data, resize_input, rgb_uncropped)
            R_gt = single_data.R
            t_gt = single_data.t

            pose = Rt_to_H(R, t)
            pose_gt = Rt_to_H(R_gt, t_gt)
            # draw the bbx with poses
            create_bounding_box(rgb_uncropped, pose, pose_gt, single_data, K)

    print('saving to ', f'predicts/bbx/{scene_id}_{img_id}.png')
    plt.imsave(f'predicts/bbx/{scene_id}_{img_id}.png', rgb_uncropped)


if __name__ == '__main__':
    test_dir = '/home/zhuolun/evalmia/arxiv/__pycache__/data/real/test'
    gpu = 0

    mspd_real = mspd_log('real', gpu)
    with open('mspd_real.txt', 'w') as f:
        json.dump(mspd_real, f)

    mspd_real_sym = mspd_log('real_sym', gpu)
    with open('mspd_real_sym.txt', 'w') as f:
        json.dump(mspd_real_sym, f)

    mspd_real_synthetic = mspd_log('real_synthetic', gpu)
    with open('mspd_real_synthetic.txt', 'w') as f:
        json.dump(mspd_real_synthetic, f)


    mspd_real_synthetic_sym = mspd_log('real_synthetic_sym', gpu)
    with open('mspd_real_synthetic_sym.txt', 'w') as f:
        json.dump(mspd_real_synthetic_sym, f)

    nocs_model_dir = '/home/zhuolun/evalmia/arxiv/__pycache__/NOCS/output/real_test_20210324T2000'

    mspd_nocs = mspd_log_nocs(nocs_model_dir, test_dir)
    with open('mspd_nocs.txt', 'w') as f:
        json.dump(mspd_nocs, f)

    draw_bbx(test_dir, 'real_sym', gpu)
