import argparse
import os
import pandas as pd
import cv2


def get_full_img_path(filename, path_dataset):
    head, tail = os.path.split(filename)
    if head == 'nightTraining':
        return os.path.join(path_dataset, 'nightTrain', 'nightTrain', tail.split('--')[0], 'frames', tail)
    if head == 'dayTraining':
        return os.path.join(path_dataset, 'dayTrain', 'dayTrain', tail.split('--')[0], 'frames', tail)
    if head == 'dayTest':
        return os.path.join(path_dataset, tail.split('--')[0], tail.split('--')[0], 'frames', tail)


def decode_csv(path_csv, path_dataset):
    n = 100
    # columns which describe bounding box
    cols = ['Annotation tag', 'Upper left corner X', 'Upper left corner Y', 'Lower right corner X',
            'Lower right corner Y']
    df = pd.read_csv(path_csv, delimiter=';').iloc[:, :6]

    # shuffle dataframe
    # df = df.sample(frac=1)
    # get first n rows
    df = df.head(n)
    # convert from string to int
    df[cols[1:]] = df[cols[1:]].astype(int)
    # group together boxes belonging to the same image
    df['Filename'] = df['Filename'].apply(lambda filename: get_full_img_path(filename, path_dataset))
    return df.groupby('Filename').apply(lambda x: [x['Filename'].iloc[0], x[cols].values.tolist()]).tolist()


def get_samples_info(path_dataset, split='train'):
    """
    :return: list of form: [[image_path, [label, x_min, y_min, x_max, y_max]'s]]
    """
    path = os.path.join(path_dataset, 'Annotations', 'Annotations')
    samples = []

    if split == 'train':
        foldrs = ['dayTrain', 'nightTrain']

        for foldr_1 in foldrs:
            path_1 = os.path.join(path, foldr_1)

            # dayClips + nightClips
            for foldr_2 in os.listdir(path_1):
                path_2 = os.path.join(path_1, foldr_2)
                # decode the annotation file
                samples.extend(decode_csv(os.path.join(path_2, 'frameAnnotationsBOX.csv'), path_dataset))

    if split == 'test':
        foldrs = ['daySequence1']

        for foldr_1 in foldrs:
            path_1 = os.path.join(path, foldr_1)
            samples.extend(decode_csv(os.path.join(path_1, 'frameAnnotationsBOX.csv'), path_dataset))

    return samples


def show_samples(samples_info, dsize, resize=False):
    for sample_info in samples_info:
        img_path, boxes = sample_info

        img = cv2.imread(img_path)
        h, w, _ = img.shape

        if resize:
            img = cv2.resize(img, dsize=(dsize[1], dsize[0]))
            hx, wx = float(dsize[0] / h), float(dsize[1] / w)

        # draw resolution if image
        resolution_string = f'{dsize[0]} x {dsize[1]}' if resize else f'{h} x {w}'
        org = (0, dsize[0]) if resize else (0, h)
        cv2.putText(img, text=resolution_string, org=org,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=(0, 255, 0), thickness=1)

        for box in boxes:
            label, x_min, y_min, x_max, y_max = box
            # rescale bounding box coordinates
            if resize:
                x_min, y_min = int(x_min * wx), int(y_min * hx)
                x_max, y_max = int(x_max * wx), int(y_max * hx)
            # draw bounding box
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max),
                          color=(0, 255, 0), thickness=1)
            # show label of bounding box
            # cv2.putText(img, text=label, org=(x_min, y_min),
            #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
            #             color=(0, 255, 0), thickness=1)
            # show resolution of bounding box
            cv2.putText(img, text=f'{x_max - x_min} x {y_max - y_min}', org=(x_min, y_min),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                        color=(0, 255, 0), thickness=1)

        cv2.imshow('Sample', img)
        cv2.waitKey(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dataset', type=str)
    args = parser.parse_args()

    samples_info = get_samples_info(path_dataset=args.path_dataset)

    show_samples(samples_info=samples_info, dsize=(480, 640), resize=False)

