from turtle import color
from models import SegDecNet
import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INPUT_WIDTH = 512  # must be the same as it was during training
INPUT_HEIGHT = 512  # must be the same as it was during training
INPUT_CHANNELS = 1  # must be the same as it was during training
dsize = INPUT_WIDTH, INPUT_HEIGHT
def plot_sample(image_name, image, segmentation, label, save_dir, decision=None, blur=True, plot_seg=False):
    plt.figure()
    plt.clf()
    # plt.subplot(1, 3, 1)
    # plt.xticks([])
    # plt.yticks([])
    # plt.title('Input image')
    # if image.shape[0] < image.shape[1]:
    #     image = np.transpose(image, axes=[1, 0, 2])
    #     segmentation = np.transpose(segmentation)
    # #     label = np.transpose(label)
    # # if image.shape[2] == 1:
    # plt.imshow(image, cmap="gray")
    # # else:
    #     # plt.imshow(image)

    # plt.subplot(1, 3, 2)
    # plt.xticks([])
    # plt.yticks([])
    # plt.title('Groundtruth')
    # plt.imshow(label, cmap="gray")

    # plt.subplot(1, 1, 1)
    plt.xticks([])
    plt.yticks([])
    # if decision is None:
    #     plt.title('Output')
    # else:
    #     plt.title(f"Score: {decision:.5f}")
    # display max
    vmax_value = max(1, np.max(segmentation))
    plt.imshow(segmentation, cmap="jet", vmax=vmax_value)
    plt.text(5,20,f"Score: {decision:.3f}",color='white')

    # plt.subplot(1, 4, 4)
    # plt.xticks([])
    # plt.yticks([])
    # plt.title('Output scaled')
    # if blur:
    #     normed = segmentation / segmentation.max()
    #     blured = cv2.blur(normed, (32, 32))
    #     plt.imshow((blured / blured.max() * 255).astype(np.uint8), cmap="jet")
    # else:
    #     plt.imshow((segmentation / segmentation.max() * 255).astype(np.uint8), cmap="jet")

    out_prefix = '{:.3f}_'.format(decision) if decision is not None else ''

    # plt.savefig(f"{save_dir}/{out_prefix}result_{image_name}.jpg", bbox_inches='tight', dpi=300)
    plt.savefig(f"{save_dir}/result_.jpg", bbox_inches=None, dpi=300,pad_inches=0)
    plt.close()

    if plot_seg:
        jet_seg = cv2.applyColorMap((segmentation * 255).astype(np.uint8), cv2.COLORMAP_JET)
        jet_seg = cv2.putText(jet_seg, f"Score: {decision:.3f}", (2, 25),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),3)
        cv2.imwrite(f"{save_dir}/segmentation_.png", jet_seg)

device = "cpu"  # cpu or cuda:IX

model = SegDecNet(device, INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS)
model.set_gradient_multipliers(0)

model_path = r"D:\\Chonnam\\Semester3\\ProjectForAI\\mixed-segdec-net-comind2021\\results\\DAGM\\N_ALL\\FOLD_6\\models\\final_state_dict.pth"
model.load_state_dict(torch.load(model_path, map_location=device))

# %%
img_path = r"D:\\Chonnam\\Semester3\\ProjectForAI\\mixed-segdec-net-comind2021\\datasets\\DAGM\\Class6\\Test\\0021.PNG"
img = cv2.imread(img_path) if INPUT_CHANNELS == 3 else cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
image = img
print(image.shape)
img = np.transpose(img, (2, 0, 1)) if INPUT_CHANNELS == 3 else img[np.newaxis]
img_t = torch.from_numpy(img)[np.newaxis].float() / 255.0  # must be [BATCH_SIZE x CHANNELS x HEIGHT x WIDTH]
print(img_t.shape)
dec_out, seg_out = model(img_t)
# img_score = torch.sigmoid(dec_out)
pred_seg = torch.sigmoid(seg_out)
prediction = torch.sigmoid(dec_out)
prediction = prediction.item()
# image = img.detach().cpu().numpy()
pred_seg = pred_seg.detach().cpu().numpy()
pred_seg = cv2.resize(pred_seg[0, 0, :, :], dsize) if len(pred_seg.shape) == 4 else cv2.resize(pred_seg[0, :, :], dsize)

plot_sample(image_name=img_path, image=image, segmentation=pred_seg, label=image, save_dir=r"./demo", decision=prediction, blur=True, plot_seg=True)

print(prediction)
print(pred_seg)
