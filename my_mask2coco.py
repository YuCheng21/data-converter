from mask2coco import main as convert


path = '/home/hpds/yucheng/dataset/ds_version'
output = './output'
category = None
directory = None

table = {
    'car': ['seg', 'seg2'],
    'motorbike': ['seg', 'seg2'],
    'aeroplane': ['seg', 'seg2'],
    'bus': ['seg', 'seg2'],
    'bicycle': ['seg', 'seg2'],
}

for k, v in table.items():
    for iv in v:
        category = k
        directory = iv
        args = {
            'category': category,
            'directory': directory,
            'path': path,
            'output': output
        }
        print(args)
        convert(args)
