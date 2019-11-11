import pandas as pd
import sys
import os
import numpy as np
from shutil import rmtree
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import cv2

DATA_PATH = ''
OUTPUT_PATH = 'aflw_processed_data'


def get_annotation(lm):
    n = lm.shape[1]
    ann = []
    # for i in range(n):
    #     ann.append(f'({lm[0,i]:.1f},{lm[1,i]:.1f})')
    # return ann


def draw_landmarks(im, lm, box, cim, clm, image_name):
    fig = plt.figure(figsize=(20, 10))

    lm = np.array([lm]).reshape(-1, 3).T.astype('float')
    clm = np.array([clm]).reshape(-1, 3).T.astype('float')

    lm_ann = get_annotation(lm)
    clm_ann = get_annotation(clm)

    plt.subplot(1, 2, 1)
    plt.imshow(im)
    rect = patches.Rectangle(
        (box[0], box[1]),
        box[2],
        box[3],
        linewidth=1,
        edgecolor='r',
        facecolor='none')
    # Add the patch to the Axes
    plt.gca().add_patch(rect)
    plt.scatter(lm[0, lm[2] != 0], lm[1, lm[2] != 0], s=10, marker='.', c='r')
    for i, txt in enumerate(lm_ann):
        plt.annotate(txt, (lm[0, i], lm[1, i]))

    plt.subplot(1, 2, 2)
    plt.imshow(cim)
    plt.scatter(
        clm[0, clm[2] != 0], clm[1, clm[2] != 0], s=10, marker='.', c='r')
    for i, txt in enumerate(clm_ann):
        plt.annotate(txt, (clm[0, i], clm[1, i]))

    plt.savefig(image_name)


if __name__ == "__main__":
    if os.path.exists(OUTPUT_PATH):
        rmtree(OUTPUT_PATH)
    os.makedirs(OUTPUT_PATH)

    DATA_PATH = sys.argv[1]
    df = pd.read_csv('aflw_processed.csv')
    crop_df = df.copy()

    # k = 0
    for idx in tqdm(range(len(df))):
        filepath = df.at[idx, 'filepath']
        image = cv2.imread(os.path.join(DATA_PATH, filepath))
        crop_df.at[idx, 'filepath'] = os.path.basename(filepath)

        # Image dimensions
        image_h, image_w, _ = image.shape

        # Face rectangle coords
        x, y = df.at[idx, 'box_x'], df.at[idx, 'box_y']
        w, h = df.at[idx, 'box_w'], df.at[idx, 'box_h']

        to_size = 224
        for landm in range(1, 22):
            lm_x = df.at[idx, str(landm) + '_x']
            lm_y = df.at[idx, str(landm) + '_y']

            # Translations
            lm_x = lm_x - x
            lm_y = lm_y - y

            # Scaling
            lm_x = lm_x * (float(to_size) / w)
            lm_y = lm_y * (float(to_size) / h)

            # Set the same
            crop_df.at[idx, str(landm) + '_x'] = lm_x
            crop_df.at[idx, str(landm) + '_y'] = lm_y

            if lm_x < 0 or lm_y < 0:
                crop_df.at[idx, str(landm) + '_vis'] = 0

        # Set records straight
        if x < 0:
            x = 0
            crop_df.at[idx, 'box_x'] = 0
        if y < 0:
            y = 0
            crop_df.at[idx, 'box_y'] = 0
        if w > image_w:
            w = image_w - x
            crop_df.at[idx, 'box_w'] = w
        if h > image_h:
            h = image_h - y
            crop_df.at[idx, 'box_h'] = h

        image_rescaled = np.copy(image[y:y + h, x:x + w])
        image_rescaled = cv2.resize(
            image_rescaled, (to_size, to_size), interpolation=cv2.INTER_AREA)
        # if k < 5:
        #     draw_landmarks(image, df.iloc[idx, 5:68], (x,y,w,h), image_rescaled,
        #                    crop_df.iloc[idx, 5:68], os.path.basename(filepath))
        #     k += 1
        # else:
        #     break

        save_name = os.path.join(OUTPUT_PATH, os.path.basename(filepath))
        cv2.imwrite(save_name, image_rescaled)
    crop_df = crop_df.drop(
        columns=['im_width', 'im_height', 'box_x', 'box_y', 'box_w', 'box_h'])
    data_len = len(crop_df)
    set_label = np.array(
        int(0.7 * data_len) * [0] + (data_len - int(0.7 * data_len)) * [1])
    np.random.shuffle(set_label)
    crop_df['val'] = set_label
    crop_df.to_csv('aflw_cropped_label.csv', index=False)
