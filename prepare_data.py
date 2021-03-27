import json
from pathlib import Path
from collections import defaultdict

################################################################
p = Path('./dataset')
train_imgs = defaultdict(dict)
test_imgs = defaultdict(dict)
for cls in range(1,7):
    for model in range(1,4):
        train_imgs[cls][model] = defaultdict(dict)
        test_imgs[cls][model] = defaultdict(dict)
        rgb = [str(i).replace('\\', '/') for i in p.glob(f'{cls}/rgb_cropped/{model}/*.png')]
        idmask = [str(i).replace('\\', '/') for i in p.glob(f'{cls}/ground_truth/IDmasks/{model}/*.png')]
        xyzmask = [str(i).replace('\\', '/') for i in p.glob(f'{cls}/ground_truth/XYZmasks/{model}/*.png')]

        train_ratio = 8

        rgb_train = []
        idmask_train = []
        xyzmask_train = []

        rgb_test = []
        idmask_test = []
        xyzmask_test = []
        for i in range(len(rgb)):
            # if i % train_ratio == 0:
            #     rgb_test.append(rgb[i])
            #     idmask_test.append(idmask[i])
            #     xyzmask_test.append(xyzmask[i])
            # else:
            rgb_train.append(rgb[i])
            idmask_train.append(idmask[i])
            xyzmask_train.append(xyzmask[i])


        train_imgs[cls][model]['rgb'] = rgb_train
        train_imgs[cls][model]['idmask'] = idmask_train
        train_imgs[cls][model]['xyzmask'] = xyzmask_train

        # test_imgs[cls][model]['rgb'] = rgb_test
        # test_imgs[cls][model]['idmask'] = idmask_test
        # test_imgs[cls][model]['xyzmask'] = xyzmask_test

with open('train_imgs_all.txt', 'w') as f:
    json.dump(train_imgs, f)

# with open('test_imgs.txt', 'w') as f:
#     json.dump(test_imgs, f)


#################################################################################################
# crop the real test image
from pathlib import Path
import numpy as np
import glob
from PIL import Image
from collections import defaultdict
import json
from utils import NestedDefaultDict, write_dict_to_json
from matplotlib import pyplot as plt

root_dir = '/home/zhuolun/evalmia/arxiv/__pycache__/data/real/test'

sorted(glob.glob(root_dir + '/scene_1/*_color.png'))

crop_pose = NestedDefaultDict()
for scene_idx in range(1, 7):
    folder_dir = root_dir + f'/scene_{scene_idx}'  # /home/zhuolun/evalmia/arxiv/__pycache__/data/real/test/scene_1
    colors = sorted(glob.glob(folder_dir + '/*_color.png'))
    folder_dir = Path(folder_dir)
    cropped = folder_dir / 'cropped'
    if not cropped.exists():
        print('Creating folder at ', str(cropped))
        cropped.mkdir(parents=False)
    for color_path in colors:
        img_idx = int(color_path.split('/')[-1].split('_')[0])
        #         print(img_idx)

        img = np.asarray(Image.open(color_path))
        mask = np.asarray(Image.open(color_path.replace('color', 'mask')))
        ids = np.unique(mask)[:-1]
        height, width, _ = img.shape
        for iid in ids:
            iid = int(iid)
            coord = np.where(mask == iid)
            margin = 5
            minx = max(min(np.min(coord[0]) - margin, height), 0)
            maxx = max(min(np.max(coord[0]) + margin, height), 0)
            miny = max(min(np.min(coord[1]) - margin, width), 0)
            maxy = max(min(np.max(coord[1]) + margin, width), 0)
            size = max(maxx - minx, maxy - miny)
            centerx = (maxx + minx) / 2
            centery = (maxy + miny) / 2
            minx = int(centerx - size / 2)
            miny = int(centery - size / 2)
            maxx = int(centerx + size / 2)
            maxy = int(centery + size / 2)
            minx = max(min(minx, height), 0)
            maxx = max(min(maxx, height), 0)
            miny = max(min(miny, width), 0)
            maxy = max(min(maxy, width), 0)

            crop_pose[scene_idx][img_idx][iid]['minx'] = minx
            crop_pose[scene_idx][img_idx][iid]['miny'] = miny
            crop_pose[scene_idx][img_idx][iid]['maxx'] = maxx
            crop_pose[scene_idx][img_idx][iid]['maxy'] = maxy

            img_cropped = img[minx:maxx, miny:maxy, :]
            #             plt.imshow(img_cropped)
            #             plt.show()
            print('saving image at ', cropped / f'{img_idx:04d}_{iid}.jpg')
            plt.imsave(cropped / f'{img_idx:04d}_{iid}.jpg', img_cropped)
write_dict_to_json(crop_pose, root_dir + '/crop_pose.txt')

###############################################################################
ps = ['dataset','synthetic_ds']
train_imgs = NestedDefaultDict()
for pstr in ps:
    p = Path(pstr)
    for cls in range(1,7):
            rgb = [str(i) for i in p.glob(f'{cls}/rgb_cropped/*/*.png')]
            xyzmask = [str(i) for i in p.glob(f'{cls}/ground_truth/XYZmasks/*/*.png')]


            rgb_train = []
            xyzmask_train = []

            for i in range(len(rgb)):
                rgb_train.append(rgb[i])
                xyzmask_train.append(xyzmask[i])


            train_imgs[cls][pstr]['rgb'] = rgb_train
            train_imgs[cls][pstr]['xyzmask'] = xyzmask_train



with open('imgs_all.txt', 'w') as f:
    json.dump(train_imgs, f)
