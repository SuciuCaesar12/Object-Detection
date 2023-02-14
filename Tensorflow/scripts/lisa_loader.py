import argparse
import os
import pandas as pd
from utils import show_examples


def get_full_img_path(filename, path_dataset):
    head, tail = os.path.split(filename)
    if head == 'nightTraining':
        return os.path.join(path_dataset, 'nightTrain', 'nightTrain', tail.split('--')[0], 'frames', tail)
    if head == 'dayTraining':
        return os.path.join(path_dataset, 'dayTrain', 'dayTrain', tail.split('--')[0], 'frames', tail)
    else:
        return os.path.join(path_dataset, tail.split('--')[0], tail.split('--')[0], 'frames', tail)


def decode_csv(path_csv, path_dataset):
    n = 1000
    # columns which describe bounding box
    cols = ['Annotation tag', 'Upper left corner X', 'Upper left corner Y', 'Lower right corner X',
            'Lower right corner Y']
    df = pd.read_csv(path_csv, delimiter=';').iloc[:, :6]
    # get first n rows
    df = df.head(n)
    # convert from string to int
    df[cols[1:]] = df[cols[1:]].astype(int)
    # group together boxes belonging to the same image
    df['Filename'] = df['Filename'].apply(lambda filename: get_full_img_path(filename, path_dataset))
    return df.groupby('Filename').apply(lambda x: [x['Filename'].iloc[0], x[cols].values.tolist()]).tolist()


class LisaLoader:

    def get_examples(self, path_dataset, split='train'):
        """
        :return: list of form: [[image_path, [label, x_min, y_min, x_max, y_max]'s]]
        """
        path = os.path.join(path_dataset, 'Annotations', 'Annotations')
        examples = []

        if split == 'train':
            foldrs = ['dayTrain', 'nightTrain']
            for foldr_1 in foldrs:
                path_1 = os.path.join(path, foldr_1)
                # dayClips + nightClips
                for foldr_2 in os.listdir(path_1):
                    path_2 = os.path.join(path_1, foldr_2)
                    # decode the annotation file
                    examples.extend(decode_csv(os.path.join(path_2, 'frameAnnotationsBOX.csv'), path_dataset))

        if split == 'test':
            foldrs = ['daySequence1']
            for foldr_1 in foldrs:
                path_1 = os.path.join(path, foldr_1)
                examples.extend(decode_csv(os.path.join(path_1, 'frameAnnotationsBOX.csv'), path_dataset))

        return examples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_dataset', type=str)
    args = parser.parse_args()

    examples = LisaLoader().get_examples(path_dataset=args.path_dataset)

    show_examples(examples=examples, dsize=(640, 720), resize=True)

