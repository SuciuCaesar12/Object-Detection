from globals import FILES, LABELS
import cv2


def create_label_map():
    labels = [{'name': label, 'id': i + 1} for i, label in enumerate(LABELS)]

    with open(FILES['LABELMAP'], 'w') as f:
        for label in labels:
            f.write('item { \n')
            f.write('\tname:\'{}\'\n'.format(label['name']))
            f.write('\tid:{}\n'.format(label['id']))
            f.write('}\n')


def show_example(example, dsize, resize=False):
    show_bb, show_label, show_res_img, show_res_bb = True, True, False, False
    img_path, boxes = example

    img = cv2.imread(img_path)
    h, w, _ = img.shape

    if resize:
        img = cv2.resize(img, dsize=(dsize[1], dsize[0]))
        hx, wx = float(dsize[0] / h), float(dsize[1] / w)

    if show_res_img:
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
        if show_bb:
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max),
                          color=(0, 255, 0), thickness=1)

        # show label of bounding box
        if show_label:
            cv2.putText(img, text=label, org=(x_min, y_min),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                        color=(0, 255, 0), thickness=1)

        # show resolution of bounding box
        if show_res_bb:
            cv2.putText(img, text=f'{x_max - x_min} x {y_max - y_min}', org=(x_min, y_min),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                        color=(0, 255, 0), thickness=1)

    cv2.imshow('Example', img)
    cv2.waitKey(0)
