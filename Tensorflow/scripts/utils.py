from globals import FILES, LABELS


def create_label_map():
    labels = [{'name': label, 'id': i + 1} for i, label in enumerate(LABELS)]

    with open(FILES['LABELMAP'], 'w') as f:
        for label in labels:
            f.write('item { \n')
            f.write('\tname:\'{}\'\n'.format(label['name']))
            f.write('\tid:{}\n'.format(label['id']))
            f.write('}\n')
