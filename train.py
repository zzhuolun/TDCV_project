import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from unet import UNet
from utils import NOCSDataset, AddGaussianNoise


def train_correspondence_block(json_file, cls, gpu, synthetic, epochs=50, batch_size=64, val_ratio=0.2,
                               save_model=True, iter_print=10):
    """
    Training a UNnet for each class using real train and/or synthetic data
    Args:
        json_file: .txt file which stores the directory of the training images
        cls: the class to train on, from 1 to 6
        gpu: gpu id to use
        synthetic: whether use synthetic data or not
        epochs: number of epochs to train
        batch_size: batch size
        val_ratio: validation ratio during training
        save_model: save model or not
        iter_print: print training results per iter_print iterations

    """
    train_data = NOCSDataset(json_file, cls, synthetic=synthetic, resize=64,
                             transform=transforms.Compose([transforms.ColorJitter(brightness=(0.6, 1.4),
                                                                                  contrast=(0.8, 1.2),
                                                                                  saturation=(0.8, 1.2),
                                                                                  hue=(-0.01, 0.01)),
                                                           AddGaussianNoise(10 / 255)]))
    print('Size of trainset ', len(train_data))
    indices = list(range(len(train_data)))
    np.random.shuffle(indices)

    num_train = len(indices)
    split = int(np.floor(num_train * val_ratio))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    num_workers = 4
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               sampler=train_sampler, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                             sampler=valid_sampler, num_workers=num_workers)
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    print("device: ", f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    # architecture for correspondence block - 13 objects + backgound = 14 channels for ID masks
    correspondence_block = UNet()
    correspondence_block = correspondence_block.to(device)

    # custom loss function and optimizer
    criterion_x = nn.CrossEntropyLoss()
    criterion_y = nn.CrossEntropyLoss()
    criterion_z = nn.CrossEntropyLoss()

    # specify optimizer
    optimizer = optim.Adam(correspondence_block.parameters(), lr=3e-4, weight_decay=3e-5)

    # training loop
    val_loss_min = np.Inf
    save_path = model_save_path(cls)
    writer = SummaryWriter(save_path.parent / save_path.stem / datetime.now().strftime("%d%H%M"))

    for epoch in range(epochs):
        t0 = time.time()
        train_loss = 0
        val_loss = 0
        print("------ Epoch ", epoch, " ---------")
        correspondence_block.train()
        print("training")
        for iter, (rgb, xmask, ymask, zmask, adr_rgb) in enumerate(train_loader):

            rgb = rgb.to(device)
            xmask = xmask.to(device)
            ymask = ymask.to(device)
            zmask = zmask.to(device)

            optimizer.zero_grad()
            xmask_pred, ymask_pred, zmask_pred = correspondence_block(rgb)

            loss_x = criterion_x(xmask_pred, xmask)
            loss_y = criterion_y(ymask_pred, ymask)
            loss_z = criterion_z(zmask_pred, zmask)

            loss = loss_x + loss_y + loss_z

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if iter % iter_print == 0:
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.
                        format(epoch, iter * len(rgb), len(train_loader.dataset),
                               100. * iter / len(train_loader), loss.item()))

        correspondence_block.eval()

        print("validating")
        for rgb, xmask, ymask, zmask, _ in val_loader:
            rgb = rgb.to(device)
            xmask = xmask.to(device)
            ymask = ymask.to(device)
            zmask = zmask.to(device)

            xmask_pred, ymask_pred, zmask_pred = correspondence_block(rgb)

            loss_x = criterion_x(xmask_pred, xmask)
            loss_y = criterion_y(ymask_pred, ymask)
            loss_z = criterion_z(zmask_pred, zmask)

            loss = loss_x + loss_y + loss_z
            val_loss += loss.item()

        # calculate average losses
        train_loss = train_loss / len(train_loader.sampler)
        val_loss = val_loss / len(val_loader.sampler)
        t_end = time.time()
        print(f'{t_end - t0} seconds')
        writer.add_scalar('train loss', train_loss, epoch)
        writer.add_scalar('val loss', val_loss, epoch)
        writer.add_scalar('epoch time', t_end - t0, epoch)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, val_loss))

        # save model if validation loss has decreased
        if val_loss <= val_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                val_loss_min,
                val_loss))
            if save_model:
                torch.save(correspondence_block.state_dict(), save_path)
            val_loss_min = val_loss
    writer.close()


def model_save_path(cls):
    """
    name of trained model
    """
    if isinstance(cls, int):
        return Path('ckpt') / f'{model_path}{cls}.pt'
    else:
        cls = [str(i) for i in cls]
        return Path('ckpt') / (model_path + ''.join(cls) + '.pt')


def train_for_each_class(gpu, synthetic):
    for model in range(1, 7):
        train_correspondence_block(json_file, model, gpu=gpu, synthetic=synthetic, save_model=True)


def train_together(gpu, synthetic):
    train_correspondence_block(json_file, (1, 2, 3, 4, 5, 6), gpu=gpu, synthetic=synthetic, save_model=True,
                               iter_print=10)


if __name__ == '__main__':
    json_file = 'train_imgs.txt'
    with_synthetic = True
    model_path = 'real' # name of trained model
    gpu = 3
    # train_together()

    train_for_each_class(gpu, with_synthetic)

