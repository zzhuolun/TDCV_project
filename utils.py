import glob
import json
import math
import os
import pickle
import sys
from collections import defaultdict, Counter
from pathlib import Path
from random import randint

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.append('model_projector/')
from model_projector.datatype import Model


def _make_square(im, fill_color=(0, 0, 0, 0)):
    """
    fill in black borders to images that are non-square
    Args:
        im (PIL image): input image
        fill_color: filling color

    Returns:
        new_im (PIL image): output image
    """

    x, y = im.size
    size = max(x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im


def mksquare_reverse(h, w):
    """
    helper function to compute the offset of height and width while making a rectangular image square
    """
    size = max(h, w)
    return int((size - h) / 2), int((size - w) / 2)


class NOCSDataset(Dataset):
    """
    train dataset class
    """

    def __init__(self, json_file, cls, synthetic=True, resize=64, transform=None):
        """
        Args:
            json_file (str): path to the json file that stores the path to images
            cls: which classes (1~6) of data to use
            transform: transform for input image
        """
        self.transform = transform
        self.resize = resize
        with open(json_file, 'r') as f:
            all_imgs = json.load(f)
        if isinstance(cls, int):
            cls = (cls,)
        rgb_path = []
        xyzmask_path = []
        for i in cls:
            rgb_path.extend(all_imgs[str(i)]['dataset']['rgb'])
            xyzmask_path.extend(all_imgs[str(i)]['dataset']['xyzmask'])
            if synthetic:
                rgb_path.extend(all_imgs[str(i)]['synthetic_ds']['rgb'])
                xyzmask_path.extend(all_imgs[str(i)]['synthetic_ds']['xyzmask'])

        assert len(rgb_path) == len(xyzmask_path)
        _check = randint(0, len(rgb_path) - 1)
        assert rgb_path[_check].split('/')[-1] == xyzmask_path[_check].split('/')[-1]

        self.rgb_path_ls = rgb_path
        self.xyzmask_path_ls = xyzmask_path

    def __len__(self):
        return len(self.rgb_path_ls)

    def __getitem__(self, idx):
        rgb = Image.open(self.rgb_path_ls[idx])
        xyzmask = Image.open(self.xyzmask_path_ls[idx])
        ##
        to_tensor = transforms.ToTensor()
        rgb = _make_square(rgb)
        xyzmask = _make_square(xyzmask)
        rgb = to_tensor(rgb)
        xyzmask = to_tensor(xyzmask)

        resize = transforms.Resize(self.resize)
        rgb = resize(rgb)
        xyzmask = resize(xyzmask)

        if self.transform:
            rgb = self.transform(rgb)  # value of rgb is between 0 and 1

        assert rgb.shape == xyzmask.shape
        xyzmask = xyzmask * 255

        xmask = xyzmask[0, :, :].type(torch.int64)
        ymask = xyzmask[1, :, :].type(torch.int64)
        zmask = xyzmask[2, :, :].type(torch.int64)
        adr_rgb = self.rgb_path_ls[idx]  # "dataset/1/rgb_cropped/1/000000.png"
        return rgb, xmask, ymask, zmask, adr_rgb


class NOCSTestset(Dataset):
    """
    Test data class
    """
    def __init__(self, root_dir, json_file, cls, transform=None):
        """
        Args:
            root_dir: root director of test dataset
            json_file: .txt stores the list of image names of each class of testset
            cls: which class to use
            transform: transform for test image
        """

        with open(json_file, 'r') as f:
            all_imgs = json.load(f)
        if isinstance(cls, int):
            cls = (cls,)
        rgb_path = []
        for i in cls:
            rgb_path.extend(all_imgs[str(i)])
        self.rgb_path_ls = rgb_path
        self.root_dir = root_dir
        # self.rgb_path_ls = glob.glob(str(Path(root_dir) / 'scene_*/cropped/*.jpg'))
        self.transform = transform

    def __len__(self):
        return len(self.rgb_path_ls)

    def __getitem__(self, idx):
        adr_rgb = os.path.join(self.root_dir, self.rgb_path_ls[idx])
        rgb = Image.open(adr_rgb)
        to_tensor = transforms.ToTensor()
        rgb = _make_square(rgb)
        if self.transform:
            rgb = self.transform(rgb)
        rgb = to_tensor(rgb)
        return rgb, adr_rgb


def id_local_to_global(model_idx, cls_idx):
    return (cls_idx + 3 * (model_idx - 1))


def id_global_to_local(glob_idx):
    model_idx = math.ceil(glob_idx / 3)
    cls_idx = glob_idx - 3 * (model_idx - 1)
    return model_idx, cls_idx


class SingleData:
    """
    properties which is related to a single image (3d model, rgb address, ground truth pose, bounding bbx, etc)
    """

    def __init__(self, rgb_adr):
        """
        Args:
            rgb_adr: the address of a cropped image
        """
        self.rgb_adr = rgb_adr  # "dataset/1/rgb_cropped/3/000408.png"
        _split = self.rgb_adr.split('/')
        self.cls = _split[1]  # "1"
        self.obj_id = _split[-2]  # "3"
        self.idx = str(int(_split[-1].split('.')[0]))  # "408"
        self.name = _split[-1]  # "000408.png"
        self.adr_uncropped = _split[0] + '/' + _split[1] + '/rgb/' + _split[4]  # "dataset/1/rgb/000408.png"
        self.model_path = _split[0] + '/' + _split[1] + '/models/' + self.obj_id + '.ply'
        self.rootdir = _split[0]

    def gt_pose(self):
        """
        Returns:
        ground truth rotation and translation
        """
        gt_path = Path(self.rootdir) / self.cls / 'scene_gt.json'
        with open(gt_path, 'r') as f:
            scene_gt = json.load(f)

        if self.rootdir == 'dataset':
            assert scene_gt[self.idx]["obj_id"] == int(self.obj_id)
            R = scene_gt[self.idx]["cam_R_m2c"]
            t = scene_gt[self.idx]["cam_t_m2c"]
        else:
            gt_ls = scene_gt[self.idx]
            for obj in gt_ls:
                if obj["obj_id"] == int(self.obj_id):
                    R = obj["cam_R_m2c"]
                    t = obj["cam_t_m2c"]
                    break
        R = np.array(R)
        R = R.reshape(3, 3)
        t = np.array(t)
        t /= 1000
        t = t.reshape(3, 1)
        return R, t

    def crop_bbx(self):
        """
        Returns:
        the position of the bbx which is used to crop out the single objects
        """
        key = self.obj_id + '/' + self.name
        with open(Path(self.rootdir) / self.cls / 'crop_pos.txt', 'r') as f:
            crop_pos_dict = json.load(f)
        minx = crop_pos_dict[key]['minx']
        miny = crop_pos_dict[key]['miny']
        maxx = crop_pos_dict[key]['maxx']
        maxy = crop_pos_dict[key]['maxy']
        return minx, miny, maxx, maxy

    def get_model(self):
        """
        Returns:
        3d vertices of the model: Nx3 np.array
        """
        model = Model()
        model.load(self.model_path)
        return model.vertices

    def bbx_3d(self):
        """
        Returns:
        bb: 3d coordinates of the 8 vertices in object frame
        draw_bbx: the lines between different vertices, used to draw the box
        """
        with open(f'{self.rootdir}/cube_vertices.json', 'r') as f:
            cube_vertices = json.load(f)
        minx = cube_vertices[self.cls][self.obj_id]['minx']
        maxx = cube_vertices[self.cls][self.obj_id]['maxx']
        miny = cube_vertices[self.cls][self.obj_id]['miny']
        maxy = cube_vertices[self.cls][self.obj_id]['maxy']
        minz = cube_vertices[self.cls][self.obj_id]['minz']
        maxz = cube_vertices[self.cls][self.obj_id]['maxz']

        bb = np.array([[minx, miny, minz],
                       [maxx, miny, minz],
                       [maxx, maxy, minz],
                       [minx, maxy, minz],
                       [minx, miny, maxz],
                       [maxx, miny, maxz],
                       [maxx, maxy, maxz],
                       [minx, maxy, maxz]])

        draw_bbx = [(0, 1), (0, 3), (1, 2), (2, 3),
                    (4, 5), (4, 7), (5, 6), (6, 7),
                    (0, 4), (1, 5), (2, 6), (3, 7)]
        return bb, draw_bbx

    def get_sym(self):
        """
        Returns:
        a list of global symmetry transformations of an object
        """
        syms = [{'R': np.eye(3), 't': np.zeros((3, 1))}]
        if self.cls in ('1', '2', '4'):
            for j in range(1, 36):
                theta = 2 * np.pi / j
                R = np.array([[np.cos(theta), 0, np.sin(theta)],
                              [0, 1, 0],
                              [-np.sin(theta), 0, np.cos(theta)]])
                syms.append({'R': R, 't': np.zeros((3, 1))})
        return syms

    def intrinsics(self):
        if self.rootdir == 'dataset':
            K = np.array([591.0125, 0.0, 322.525, 0.0, 590.16775, 244.11084, 0.0, 0.0, 1.0])
        else:
            K = np.array([888.888916015625, 0.0, 320.0, 0.0, 1000.0, 240.0, 0.0, 0.0, 1.0])
        return K.reshape((3, 3))


class SingleTestdata():
    """
    The counterpart of Singedata for test images
    """
    def __init__(self, rgb_adr):
        """
        Args:
            rgb_adr: the address of a cropped image
        """

        self.rgb_adr = rgb_adr  # address of the cropped image, e.g data/real/test/scene_1/cropped/0000_2.jpg
        rgb_adr = Path(self.rgb_adr)
        img_idx = rgb_adr.stem.split('_')[0]  # '0000'
        self.img_idx = int(img_idx)  # 0
        self.inst_id = int(rgb_adr.stem.split('_')[-1])  # 2
        self.adr_uncropped = str(rgb_adr.parents[1] / (
                img_idx + '_color.png'))  # 'data/real/test/scene_1/0000_color.png'

        self.scene_idx = int(str(rgb_adr.parents[1]).split('_')[-1])  # 1
        gt = rgb_adr.parents[2] / 'gts' / f'results_real_test_scene_{self.scene_idx}_{img_idx}.pkl'
        with open(gt, 'rb') as f:
            gt_data = pickle.load(f)
        R = gt_data['gt_RTs'][self.inst_id - 1][:3, :3]
        self.R = R / np.sqrt((R.T @ R)[0][0])
        self.t = gt_data['gt_RTs'][self.inst_id - 1][:3, 3].reshape(3, 1)
        # self.meta_path = str(rgb_adr.parents[1] / (img_idx + '_meta.txt'))  # 'data/real/test/scene_1/0000_meta.txt'
        self.cls_id, objname = self._cls_objname(
            str(rgb_adr.parents[1] / (img_idx + '_meta.txt')))  # read the class id and the name of the object
        self.vertices, _ = load_mesh(rgb_adr.parents[2] / 'obj_models' / 'real_test' / (objname + '.obj'))
        self.adr_coord = str(rgb_adr.parents[1] / (
                img_idx + '_coord.png'))

    def _cls_objname(self, meta_path):
        with open(meta_path, 'r') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            words = line[:-1].split(' ')
            inst_id = int(words[0])
            if int(inst_id) == self.inst_id:
                return int(words[1]), words[-1]

    def crop_bbx(self):
        """
         Returns:
         the position of the bbx which is used to crop out the single objects
         """
        with open(Path(self.rgb_adr).parents[2] / 'crop_pose.txt', 'r') as f:
            crop_bbx = json.load(f)
        minx = crop_bbx[str(self.scene_idx)][str(self.img_idx)][str(self.inst_id)]['minx']
        miny = crop_bbx[str(self.scene_idx)][str(self.img_idx)][str(self.inst_id)]['miny']
        maxx = crop_bbx[str(self.scene_idx)][str(self.img_idx)][str(self.inst_id)]['maxx']
        maxy = crop_bbx[str(self.scene_idx)][str(self.img_idx)][str(self.inst_id)]['maxy']
        return minx, miny, maxx, maxy

    def get_sym(self):
        """
        Returns:
        a list of global symmetry transformations of an object
        """
        syms = [{'R': np.eye(3), 't': np.zeros((3, 1))}]
        if self.cls_id in (1, 2, 4):
            for j in range(1, 36):
                theta = 2 * np.pi / j
                R = np.array([[np.cos(theta), 0, np.sin(theta)],
                              [0, 1, 0],
                              [-np.sin(theta), 0, np.cos(theta)]])
                syms.append({'R': R, 't': np.zeros((3, 1))})
        return syms

    def bbx_3d(self):
        """
        Returns:
        bb: 3d coordinates of the 8 vertices in object frame
        draw_bbx: the lines between different vertices, used to draw the box
        """
        minx, miny, minz = self.vertices.min(axis=0)
        maxx, maxy, maxz = self.vertices.max(axis=0)

        bb = np.array([[minx, miny, minz],
                       [maxx, miny, minz],
                       [maxx, maxy, minz],
                       [minx, maxy, minz],
                       [minx, miny, maxz],
                       [maxx, miny, maxz],
                       [maxx, maxy, maxz],
                       [minx, maxy, maxz]])

        draw_bbx = [(0, 1), (0, 3), (1, 2), (2, 3),
                    (4, 5), (4, 7), (5, 6), (6, 7),
                    (0, 4), (1, 5), (2, 6), (3, 7)]
        return bb, draw_bbx


def project_pts(pts, K, R, t):
    """
    Projects 3D points.
    Args:
        pts: nx3 ndarray with the 3D points.
        K: 3x3 ndarray with an intrinsic camera matrix.
        R: 3x3 ndarray with a rotation matrix.
        t: 3x1 ndarray with a translation vector.

    Returns:
        nx2 ndarray with 2D image coordinates of the projections.
    """
    assert (pts.shape[1] == 3)
    P = K.dot(np.hstack((R, t)))
    pts_h = np.hstack((pts, np.ones((pts.shape[0], 1))))
    pts_im = P.dot(pts_h.T)
    pts_im /= pts_im[2, :]
    return pts_im[:2, :].T


def mspd(R_est, t_est, R_gt, t_gt, K, pts, syms):
    """
    Maximum Symmetry-Aware Projection Distance (MSPD). See: http://bop.felk.cvut.cz/challenges/bop-challenge-2019/
    Args:
        R_est: 3x3 ndarray with the estimated rotation matrix.
        t_est: 3x1 ndarray with the estimated translation vector.
        R_gt: 3x3 ndarray with the ground-truth rotation matrix.
        t_gt: 3x1 ndarray with the ground-truth translation vector.
        K: 3x3 ndarray with the intrinsic camera matrix.
        pts: nx3 ndarray with 3D model points.
        syms: Set of symmetry transformations, each given by a dictionary with:
            - 'R': 3x3 ndarray with the rotation matrix.
            - 't': 3x1 ndarray with the translation vector.
    Returns:
        The calculated error, and the index of the symmetric transform
    """
    proj_est = project_pts(pts, K, R_est, t_est)
    es = []
    for sym in syms:
        R_gt_sym = R_gt.dot(sym['R'])
        t_gt_sym = R_gt.dot(sym['t']) + t_gt
        proj_gt_sym = project_pts(pts, K, R_gt_sym, t_gt_sym)
        es.append(np.linalg.norm(proj_est - proj_gt_sym, axis=1).max())
    return min(es), es.index(min(es))


class NestedDefaultDict(defaultdict):
    """
    A helper class, defines a dictionary that could insert keys at any level
    """
    def __init__(self, *args, **kwargs):
        super(NestedDefaultDict, self).__init__(NestedDefaultDict, *args, **kwargs)

    def __repr__(self):
        return repr(dict(self))


def write_dict_to_json(dic, json_path):
    with open(json_path, 'w') as f:
        json.dump(dic, f)


def load_mesh(mesh_path, is_save=False, is_normalized=False, is_flipped=False):
    with open(mesh_path, 'r') as f:
        lines = f.readlines()

    vertices = []
    faces = []
    for l in lines:
        l = l.strip()
        words = l.split(' ')
        if words[0] == 'v':
            vertices.append([float(words[1]), float(words[2]), float(words[3])])
        if words[0] == 'f':
            face_words = [x.split('/')[0] for x in words]
            faces.append([int(face_words[1]) - 1, int(face_words[2]) - 1, int(face_words[3]) - 1])

    vertices = np.array(vertices, dtype=np.float64)
    # flip mesh to unity rendering
    if is_flipped:
        vertices[:, 2] = -vertices[:, 2]
    faces = np.array(faces, dtype=np.int32)

    if is_normalized:
        maxs = np.amax(vertices, axis=0)
        mins = np.amin(vertices, axis=0)
        diffs = maxs - mins
        assert diffs.shape[0] == 3
        vertices = vertices / np.linalg.norm(diffs)

    if is_save:
        np.savetxt(mesh_path.replace('.obj', '_vertices.txt'), X=vertices)

    return vertices, faces


class AddGaussianNoise(object):
    def __init__(self, std=5 / 255, mean=0.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return torch.clamp((tensor + torch.randn(tensor.size()) * self.std + self.mean), 0, 1)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def list_duplicates_of(seq, item):
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item, start_at + 1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs


def align_pred_gt(pred_data, inst_id):
    """
    Align the ground truth pose and the predicted pose in the output of NOCS implementation.
    Since the predicted pose idx are not same as the ground truth pose in the output of NOCS, alignment has to be done to read in the ground truth pose of the corresponding predicted pose
    Args:
        pred_data: the dictionary which stores the output information of NOCS
        inst_id: the instance id of the object in the whole scene

    Returns:
        the aligned id of the pose, i.e. the id of the predicted pose of the instance
    """

    cls = pred_data['gt_class_ids'][inst_id - 1]
    gt_bbx = pred_data['gt_bboxes'][inst_id - 1]
    center_x = (gt_bbx[0] + gt_bbx[2]) / 2
    center_y = (gt_bbx[1] + gt_bbx[3]) / 2
    pred_id_candidates = list_duplicates_of(pred_data['pred_class_ids'].tolist(), cls)
    if len(pred_id_candidates) == 1:
        aligned_id = pred_id_candidates[0]
    elif len(pred_id_candidates) == 0:
        dist = []
        for i in range(len(pred_data['pred_class_ids'])):
            pred_bbx = pred_data['pred_bboxes'][i]
            center_x_pred = (pred_bbx[0] + pred_bbx[2]) / 2
            center_y_pred = (pred_bbx[1] + pred_bbx[3]) / 2
            dist.append(np.sqrt((center_x_pred - center_x) ** 2 + (center_y_pred - center_y) ** 2))
        dist_min = min(dist)
        if dist_min > 16 and len(pred_data['pred_class_ids']) < len(pred_data['gt_class_ids']):
            return False
        else:
            aligned_id = dist.index(min(dist))
    else:
        dist = []
        for id_can in pred_id_candidates:
            pred_bbx = pred_data['pred_bboxes'][id_can]
            center_x_pred = (pred_bbx[0] + pred_bbx[2]) / 2
            center_y_pred = (pred_bbx[1] + pred_bbx[3]) / 2
            dist.append(np.sqrt((center_x_pred - center_x) ** 2 + (center_y_pred - center_y) ** 2))
        aligned_id = pred_id_candidates[dist.index(min(dist))]

    center_x_pred = (pred_data['pred_bboxes'][aligned_id][0] + pred_data['pred_bboxes'][aligned_id][2]) / 2
    center_y_pred = (pred_data['pred_bboxes'][aligned_id][1] + pred_data['pred_bboxes'][aligned_id][3]) / 2
    if np.sqrt((center_x_pred - center_x) ** 2 + (center_y_pred - center_y) ** 2) >= 20:
        return False
    else:
        return aligned_id

def render_gt_poses(cfg, dataset_id, seqs2load):
    """
    render the xyzmask and crop out single object from the whole scene
    Args:
        cfg: config data from config.ini
        dataset_id: the id of classes, e.g. [1,2,3,4,5,6]
        seqs2load: the id of 3d model .ply file, e.g ['1', '2', '3']
    """
    print("Collecting real samples:")
    ren = Renderer(cfg)

    path_seq = util.read_cfg_string(cfg, 'dataset', 'path', default=None)
    path_seq = Path(path_seq) / str(dataset_id)
    dataset_dir_structure(dataset_id, path_seq)
    names2models = load_models(path_seq / 'models', seqs2load)
    gt, cameras = load_gt(path_seq)  # gt: camera extrinsics, i.e. R, T; cameras: camera intrinsics

    crop_pos = defaultdict(dict)

    for i, img_id in enumerate(gt.keys()):
        try:
            if i % 50 == 0:
                print('{}/{}'.format(i, len(gt.keys())))
            gt_i, camera_i = gt[img_id], cameras[img_id]

            cam = np.array(camera_i['cam_K']).reshape((3, 3))
            ren.set_cam(cam)

            R = np.array(gt_i['cam_R_m2c']).reshape((3, 3))
            t = np.array(gt_i['cam_t_m2c'])
            if dataset_id in (1, 2, 4):
                R = symmetric_rot(R, t)
            pose = np.empty((4, 4))
            pose[:3, :3] = R  # * scale
            pose[:3, 3] = t * 0.001
            pose[3] = [0, 0, 0, 1]
            ren.clear()
            ren.draw_model(names2models[gt_i['obj_id']], pose)
            rendered_nocs, rendered_depth = ren.finish()
            rendered_nocs = rendered_nocs[:, :, :3]

            if not np.any(rendered_depth):
                continue
            height = int(cfg['renderer']['resolution'].split(',')[1])
            width = int(cfg['renderer']['resolution'].split(',')[0])

            mask = np.where(rendered_depth != 0)
            margin = 5
            minx = max(min(np.min(mask[0]) - margin, height), 0)
            maxx = max(min(np.max(mask[0]) + margin, height), 0)
            miny = max(min(np.min(mask[1]) - margin, width), 0)
            maxy = max(min(np.max(mask[1]) + margin, width), 0)
            minx, miny, maxx, maxy = mk_square(minx, miny, maxx, maxy, height, width)
            crop_pos[str(gt_i['obj_id']) + '/' + f"{int(img_id):06d}.png"]['minx'] = minx
            crop_pos[str(gt_i['obj_id']) + '/' + f"{int(img_id):06d}.png"]['miny'] = miny
            crop_pos[str(gt_i['obj_id']) + '/' + f"{int(img_id):06d}.png"]['maxx'] = maxx
            crop_pos[str(gt_i['obj_id']) + '/' + f"{int(img_id):06d}.png"]['maxy'] = maxy

            rendered_nocs_cropped = rendered_nocs[minx:maxx, miny:maxy, :]
            rendered_depth_cropped = rendered_depth[minx:maxx, miny:maxy, :]
            rgb_path = path_seq / 'rgb' / f"{int(img_id):06d}.png"
            rgb = plt.imread(rgb_path)
            rgb = rgb[minx:maxx, miny:maxy, :]

            model_cls = float(path_seq.name)
            ID_mask = np.zeros_like(rendered_depth_cropped)
            ID_mask[np.where(rendered_depth_cropped != 0)] = (gt_i['obj_id'] + 3 * (model_cls-1)) / 255

            idmask_path = path_seq / 'ground_truth' / "IDmasks" / str(gt_i['obj_id']) / f"{int(img_id):06d}.png"
            xyzmask_path = path_seq / 'ground_truth' / "XYZmasks" / str(gt_i['obj_id']) / f"{int(img_id):06d}.png"
            rgb_path = path_seq / 'rgb_cropped' / str(gt_i['obj_id']) / f"{int(img_id):06d}.png"

            mpimg.imsave(idmask_path, np.dstack([ID_mask, ID_mask, ID_mask]))
            mpimg.imsave(xyzmask_path, rendered_nocs_cropped)
            mpimg.imsave(rgb_path, rgb)

        except Exception as e:
            print(e)
            continue

    with open(path_seq / 'crop_pos.txt','w') as f:
        json.dump(crop_pos, f)


def dataset_dir_structure(dataset_id, root_dir):
    # idmask_path = Path(root_dir) / 'ground_truth' / "IDmasks"
    xyzmask_path = Path(root_dir) / 'ground_truth' / "XYZmasks"
    rgb_path = Path(root_dir) / 'rgb_cropped'
    # path_ls = [idmask_path, xyzmask_path, rgb_path]
    path_ls = [xyzmask_path, rgb_path]
    for cls in range(1, 7):
        for dir_item in path_ls:
            p = dir_item / str(cls)
            if not p.exists():
                print('Creating folder at ', str(p))
                p.mkdir(parents=True)

def mk_square(minx, miny, maxx, maxy, height, width):
    centerx = (minx+maxx)/2
    centery = (miny+maxy)/2
    l = max(maxx-minx, maxy-miny)/2

    minx = int(max(min(centerx - l, height), 0))
    maxx = int(max(min(centerx + l , height), 0))
    miny = int(max(min(centery - l, width), 0))
    maxy = int(max(min(centery + l, width), 0))

    return minx, miny, maxx, maxy


def symmetric_rot(gt_rotation, gt_translation):
    """
    Remap the rotation for symmetric objects
    Args:
        gt_rotation: original ground truth rotation
        gt_translation: original ground truth translation

    Returns:
        remapped rotation matrix
    """
    camera_location = -np.matmul(np.linalg.inv(gt_rotation), gt_translation)
    camera_location_zx = camera_location[[2, 0]]
    camera_location_zx = camera_location_zx / np.linalg.norm(camera_location_zx)

    symmetry_direction_zx = np.array((1, 0))

    cos = np.dot(camera_location_zx, symmetry_direction_zx)
    angle = np.arccos(cos)

    direction = 1. if np.cross(camera_location_zx, symmetry_direction_zx) > 0 else -1

    rotation_y = Rotation.from_euler('y', direction * angle).as_matrix()

    gt_rotation = np.matmul(gt_rotation, rotation_y.T)
    return gt_rotation
