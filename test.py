import os
import torch
import cv2
from PIL import Image
import numpy as np
import imageio

from model.fcn8s import FCN8s
from utils.metrics import label_accuracy_score

test_folder = "./Test_data/imgs"
trained_model = "./best_model/fcn_cvppp_14.pth"
gt_folder = "./Test_data/lbls"
output_folder = "./model_outputs"

mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

def img_transform(img):
        img = img[:, :, ::-1]   # RGB -> BGR
        img = img.astype(np.float64)
        img -= mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        return img

def img_untransform(img):
    img = img.numpy()
    img = img.transpose(1, 2, 0)
    img += mean_bgr
    img = img.astype(np.uint8)
    img = img[:, :, ::-1]   # BGR -> RGB
    return img

def main():
    
    model = FCN8s(n_class = 2)

    overall_mean_iu = 0

    test_imgs = os.listdir(test_folder)
    test_lbls = os.listdir(gt_folder)

    test_imgs.sort()
    test_lbls.sort()

    for imgfile, lblfile in zip(test_imgs, test_lbls):
        img = Image.open(os.path.join(test_folder,imgfile))
        gt_label = Image.open(os.path.join(gt_folder, lblfile))

        img = np.array(img, dtype=np.uint8)
        gt_label = np.array(gt_label, dtype=np.uint8)

        img = cv2.resize(img, (410,410), interpolation = cv2.INTER_AREA)
        gt_label = cv2.resize(gt_label, (410,410), interpolation = cv2.INTER_NEAREST)
        gt_label //= 255

        img = img_transform(img)  
        img = torch.unsqueeze(img, dim = 0)

        model.load_state_dict(torch.load(trained_model)) 
        out_score = model(img)
        pred, idxs = torch.max(out_score, dim=1)
        out_lbl = idxs.detach().cpu().numpy()
        out_lbl = out_lbl.transpose(1, 2, 0)
        out_lbl = out_lbl.astype(np.uint8)

        #if not os.path.exists(output_folder):
            #os.makedirs(output_folder)
        #imageio.imwrite(os.path.join(output_folder, lblfile), out_lbl*255)
        _ , _ , mean_iu , _ = label_accuracy_score(gt_label, out_lbl, 2)
        print(mean_iu)
        overall_mean_iu += mean_iu

    print(overall_mean_iu/7)

if __name__ == "__main__":
    main()
