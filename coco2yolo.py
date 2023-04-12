import logging
from pathlib import Path
import argparse

from pylabel import importer


def main(args: dict):
    logging.getLogger().setLevel(logging.CRITICAL)

    dataset_name = args['name']
    category = args['category']
    num = args['num']

    # Specify path to the coco.json file
    path_to_annotations = f"{args['path_anno']}/{category}_seg{num}.json"
    # Specify the path to the images (if they are in a different folder than the annotations)
    path_to_images = f"{args['path_img']}/{category}/merge_image{num}"

    # Import the dataset into the pylable schema
    dataset = importer.ImportCoco(path_to_annotations, path_to_images=path_to_images, name=f'{dataset_name}_{category}')

    dataset.splitter.StratifiedGroupShuffleSplit(train_pct=.8, val_pct=.1, test_pct=.1, batch_size=500)

    output_path = Path(f'{args["output"]}/{dataset_name}/{category}/seg{num}/labels')
    output_path.mkdir(parents=True, exist_ok=True)

    dataset.export.ExportToYoloV5(
        output_path=output_path,
        yaml_file='dataset.yaml',
        copy_images=True,
        use_splits=True,
        segmentation=True
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert dataset to yolo from coco format')
    parser.add_argument('--name', '-n', type=str, default='uda_part', help='name')
    parser.add_argument('--path_anno', '-pa', type=str, default='/home/hpds/yucheng/data-converter/output', help='path_anno')
    parser.add_argument('--path_img', '-pi', type=str, default='/home/hpds/yucheng/dataset/ds_version', help='path_img')
    parser.add_argument('--category', '-c', type=str, default='car', help='category')
    parser.add_argument('--num', '-num', type=str, default='', help='num')
    parser.add_argument('--output', '-o', type=str, default='./output', help='output')
    args = parser.parse_args()
    # > directory=image && mkdir merge_$directory; find $PWD/$directory -name "*.png" | xargs -i ln -sf {} $PWD/merge_$directory
    main(vars(args))
