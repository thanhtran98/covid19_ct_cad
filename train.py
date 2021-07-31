from models.segment_models import Unet
from models.classify_models import get_classify_model
from data.dataloader import get_classify_loader, get_segment_loader
from utils.post_process import gen_mask
import argparse
import torch
import torch.nn as nn
from os.path import join
import os
import shutil
import time
from tqdm import tqdm


def train(data_dir, label_dir, segment_ckp, log_dir, epochs=160, img_size=(192, 288), batch_size=16, thresh_obj=256):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ----------------------------------- Segmentation stage -----------------------------------------
    covid_dir = join(data_dir, 'CT_COVID')
    noncovid_dir = join(data_dir, 'CT_NonCOVID')
    lung_loader = get_segment_loader(
        (covid_dir, noncovid_dir), img_size=(256, 256), batch_size=8)

    segment_model = Unet(n_class=1, n_channel=1, norm='mvn')
    segment_model.to(device)
    segment_model.load_state_dict(torch.load(segment_ckp))

    gen_mask(segment_model, lung_loader, img_size, device, thresh_obj=thresh_obj)

    # ---------------------------------- Classification stage -----------------------------------------
    covid_dir = join(data_dir, 'CT_COVID_mask')
    noncovid_dir = join(data_dir, 'CT_NonCOVID_mask')

    train_label_pos = join(label_dir, 'COVID', 'trainCT_COVID.txt')
    val_label_pos = join(label_dir, 'COVID', 'valCT_COVID.txt')
    train_label_neg = join(label_dir, 'NonCOVID', 'trainCT_NonCOVID.txt')
    val_label_neg = join(label_dir, 'NonCOVID', 'valCT_NonCOVID.txt')

    train_loader = get_classify_loader((noncovid_dir, covid_dir), (
        train_label_neg, train_label_pos), mode='train', image_size=img_size, batch_size=batch_size)
    val_loader = get_classify_loader((noncovid_dir, covid_dir), (
        val_label_neg, val_label_pos), mode='val', image_size=img_size, batch_size=batch_size)

    model = get_classify_model()
    model.to(device)

    # --------------------------------- Training process -----------------------------------------------
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    # lr_sch = torch.optim.lr_scheduler.StepLR(optimizer, 80, 3/10)

    best_acc = 0.0
    os.makedirs(log_dir, exist_ok=True)
    loader_dict = {
        'train': train_loader,
        'val': val_loader
    }
    modes = ['train', 'val']

    for epoch in range(epochs):

        s = "Epoch [{}/{}]:".format(epoch+1, epochs)
        acc_dict = {}
        start = time.time()
        for mode in modes:

            running_acc = 0.0
            running_loss = 0.0
            ova_len = loader_dict[mode].dataset.n_data
            if mode == 'train':
                model.train()
            else:
                model.eval()
            for i, data in enumerate(tqdm(loader_dict[mode])):

                imgs, labels = data[0].to(device), data[1].to(device)
                preds = model(imgs)
                loss = criterion(preds, labels)
                if mode == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                iter_len = imgs.size()[0]
                preds = (preds > 0.5).float()
                running_acc += (preds == labels).float().sum()
                running_loss += loss.item()*iter_len
            running_acc /= ova_len
            acc_dict[mode] = running_acc
            running_loss /= ova_len
            s += " {}_acc {:.4f} - {}_loss {:.3f} -".format(
                mode, running_acc, mode, running_loss)
        end = time.time()
        s = s[:-1] + "({:.1f})s".format(end-start)
        print(s)
        if acc_dict['val'] >= best_acc:
            best_acc = running_acc
            torch.save(model.state_dict(), os.path.join(log_dir, 'best.pt'))
            print('new checkpoint saved!')
        # lr_sch.step()
        # print('current lr: {:.4f}'.format(lr_sch.get_lr()[0]))


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='',
                        help='directory to the folder contain images')
    parser.add_argument('--label', type=str, default='',
                        help='directory to the folder contain all labels of')
    parser.add_argument('--log_dir', type=str,
                        default='./checkpoint', help='logging directory')
    parser.add_argument('--segment-ckp', type=str,
                        help='path to the pretrained weight of segmentation model')
    parser.add_argument('--epochs', type=int, default=160)
    parser.add_argument('--batch-size', type=int, default=16,
                        help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int,
                        default=256, help='train, val image size (pixels)')
    parser.add_argument('--thresh-obj', type=int,
                        default=256, help='object threshold')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    opt.img_size = tuple(opt.img_size) if isinstance(
        opt.img_size, list) else (opt.img_size, opt.img_size)

    train(opt.data, opt.label, opt.segment_ckp, opt.log_dir,
          opt.epochs, opt.img_size, opt.batch_size, opt.thresh_obj)
