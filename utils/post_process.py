from skimage.morphology import remove_small_objects, remove_small_holes
import os
import numpy as np
import shutil
import torch
import cv2
from tqdm import tqdm

def get_id2max(t_list):
    max_1 = np.argmax(t_list)
    # print(max_1, max_1.shape)
    if isinstance(max_1,np.int64):
        t_list[max_1] = 0
        max_2 = np.argmax(t_list)
        return max_1, max_2
    else:
        return max_1[0], max_1[1]

def area(vs):
    a = 0
    x0,y0 = vs[0]
    for [x1,y1] in vs[1:]:
        dx = x1-x0
        dy = y1-y0
        a += 0.5*(y0*dx - x0*dy)
        x0 = x1
        y0 = y1

    return a

def get_areamax2(contours):
    area_max = 0.0
    area_amax = 0.0
    id_max = 0
    id_amax = 0
    for i, contour in enumerate(contours):
        contour = contour.reshape((-1,2))
        s = area(contour)
        if s > area_max:
            area_amax = area_max
            area_max = s
            id_amax = id_max
            id_max = i
        elif s > area_amax:
            area_amax = s
            id_amax = i
        
    return id_max, id_amax

def gen_mask(model, loader, img_size, device, thresh_val=0.5, thresh_obj = 100, thresh_hole = 36, use_contour=True, add_pixel=20):
    model.eval()
    for folder_mask in loader.dataset.folders_mask:
        if os.path.exists(folder_mask):
            shutil.rmtree(folder_mask)
        os.mkdir(folder_mask)

    with torch.no_grad() as tng:
        for i, data in enumerate(tqdm(loader)):
            imgs, paths = data[0].to(device), data[1]
            preds = model(imgs[:,0:1,:,:])
            preds = preds.cpu().data.numpy()
            imgs = imgs.cpu().data.numpy()
            # imgs = imgs[:,0,:,:]
            imgs = np.stack([imgs[:,0,:,:], imgs[:,1,:,:], imgs[:,2,:,:]], axis=-1)
            preds = preds[:,0,:,:] > thresh_val
            for j, mask in enumerate(preds):
                remove_small_objects(mask, thresh_obj, in_place=True)
                remove_small_holes(mask, thresh_hole, in_place=True)
                if use_contour:
                    contours, _ = cv2.findContours((mask*255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    # list_len = [aa.shape[0] for aa in contours]
                    # ms = get_id2max(list_len)
                    ms = get_areamax2(contours)
                    new_img = np.zeros(mask.shape).astype(np.uint8)
                    [cv2.fillPoly(new_img, [contours[aa].reshape((-1,2))], (255,255,255)) for aa in ms]
                    mask = new_img/255.
                else:
                    mask = mask*1.0
                h, w = mask.shape
                y_axis, x_axis= np.where(mask == 1.0)
                y_min = np.min(y_axis)
                y_min = y_min - add_pixel if y_min >= add_pixel else 0
                y_max = np.max(y_axis)
                y_max = y_max + add_pixel if y_max < h - add_pixel else h-1
                x_min = np.min(x_axis)
                x_min = x_min - add_pixel if x_min >= add_pixel else 0
                x_max = np.max(x_axis)
                x_max = x_max + add_pixel if x_max < w - add_pixel else w-1
                img_new = imgs[j][y_min:y_max, x_min:x_max]
                img_new = cv2.resize((255*img_new).astype(np.uint8), (img_size[1], img_size[0]))
                img_base_dir = os.path.dirname(paths[j])
                img_name = os.path.basename(paths[j])
                save_path = os.path.join(img_base_dir+'_mask', img_name)
                cv2.imwrite(save_path, img_new)