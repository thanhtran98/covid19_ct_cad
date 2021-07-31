from models.segment_models import Unet
from models.classify_models import get_classify_model
from data.dataloader import get_classify_loader, get_segment_loader
from utils.post_process import gen_mask
from utils.metrics import report_result
import argparse
import torch
import torch.nn as nn
from os.path import join
import numpy as np
from tqdm import tqdm


def make_prediction(model, loader, device):
    model.eval()
    with torch.no_grad() as tng:
        for i, data in enumerate(tqdm(loader)):
            imgs, targets = data[0].to(device), data[1].to(device)
            preds = model(imgs)
            if i:
                targets_sum = np.concatenate(
                    [targets_sum, targets.cpu().numpy()], axis=0)
                preds_sum = np.concatenate(
                    [preds_sum, preds.cpu().numpy()], axis=0)
            else:
                targets_sum = targets.cpu().numpy()
                preds_sum = preds.cpu().numpy()

    return targets_sum, preds_sum


def test(data_dir, label_dir, classify_ckp, segment_ckp, img_size=(192, 288), batch_size=16, thresh_obj=256):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ----------------------------------- Segmentation stage -----------------------------------------
    # covid_dir = join(data_dir, 'CT_COVID')
    # noncovid_dir = join(data_dir, 'CT_NonCOVID')
    # lung_loader = get_segment_loader(
    #     (covid_dir, noncovid_dir), img_size=(256, 256), batch_size=8)

    # segment_model = Unet(n_class=1, n_channel=1, norm='mvn')
    # segment_model.to(device)
    # segment_model.load_state_dict(torch.load(segment_ckp))

    # gen_mask(segment_model, lung_loader, img_size,
    #          device, thresh_obj=thresh_obj)

    # ---------------------------------- Classification stage -----------------------------------------
    covid_dir = join(data_dir, 'CT_COVID_mask')
    noncovid_dir = join(data_dir, 'CT_NonCOVID_mask')

    test_label_pos = join(label_dir, 'COVID', 'testCT_COVID.txt')
    val_label_pos = join(label_dir, 'COVID', 'valCT_COVID.txt')
    test_label_neg = join(label_dir, 'NonCOVID', 'testCT_NonCOVID.txt')
    val_label_neg = join(label_dir, 'NonCOVID', 'valCT_NonCOVID.txt')

    test_loader = get_classify_loader((noncovid_dir, covid_dir), (
        test_label_neg, test_label_pos), mode='test', image_size=img_size, batch_size=batch_size)
    val_loader = get_classify_loader((noncovid_dir, covid_dir), (
        val_label_neg, val_label_pos), mode='val', image_size=img_size, batch_size=batch_size)

    model = get_classify_model()
    model.to(device)
    model.load_state_dict(torch.load(classify_ckp))

    # ------------------------------------- Validation ------------------------------------------------
    y_g, y_p = make_prediction(model, val_loader, device)
    print('Result on Validation set:', '\n')
    report_result(y_g, y_p)

    y_g, y_p = make_prediction(model, test_loader, device)
    print('Result on test set:', '\n')
    report_result(y_g, y_p)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='',
                        help='directory to the folder contain images')
    parser.add_argument('--label', type=str, default='',
                        help='directory to the folder contain all labels of')
    parser.add_argument('--classify-ckp', type=str,
                        help='path to the trained weight of classification model')
    parser.add_argument('--segment-ckp', type=str,
                        help='path to the pretrained weight of segmentation model')
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

    test(opt.data, opt.label, opt.classify_ckp, opt.segment_ckp,
         opt.img_size, opt.batch_size, opt.thresh_obj)