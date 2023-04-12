from coco2yolo import main as convert

name = 'uda_part'
path_anno = '/home/hpds/yucheng/data-converter/output'
path_img = '/home/hpds/yucheng/dataset/ds_version'
output = './output'
category = None
num = None

table = {
    'car': ['', '2'],
    'motorbike': ['', '2'],
    'aeroplane': ['', '2'],
    'aeroplane': ['2'],
    'bus': ['', '2'],
    'bicycle': ['', '2'],
}

for k, v in table.items():
    for iv in v:
        category = k
        num = iv
        args = {
            'name': name,
            'path_anno': path_anno,
            'path_img': path_img,
            'category': category,
            'num': num,
            'output': output
        }
        print(args)
        convert(args)
