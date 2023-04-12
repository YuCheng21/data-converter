import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path
import argparse

import numpy as np
from PIL import Image
from shapely.geometry import MultiPolygon, Polygon
from skimage import measure
from tqdm import tqdm


def create_sub_masks(mask_image):
    width, height = mask_image.size

    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x, y))  # [:3]

            # # If the pixel is not black...
            # if pixel != (0, 0, 0):
            # Check to see if we've created a sub-mask...
            pixel_str = pixel
            sub_mask = sub_masks.get(pixel_str)
            if sub_mask is None:
                # Create a sub-mask (one bit per pixel) and add to the dictionary
                # Note: we add 1 pixel of padding in each direction
                # because the contours module doesn't handle cases
                # where pixels bleed to the edge of the image
                sub_masks[pixel_str] = Image.new('1', (width+2, height+2))

            # Set the pixel value to 1 (default is 0), accounting for padding
            sub_masks[pixel_str].putpixel((x+1, y+1), 1)

    return sub_masks


def create_sub_mask_annotation(sub_mask, image_id, category_id, is_crowd):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)
    contours = measure.find_contours(np.array(sub_mask), 0.5, positive_orientation='low')

    segmentations = []
    polygons = []
    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        for i in range(len(contour)):
            row, col = contour[i]
            contour[i] = (col - 1, row - 1)
        # Make a polygon and simplify it
        poly = Polygon(contour)
        poly = poly.simplify(1.0, preserve_topology=False)
        if (poly.geom_type == 'MultiPolygon'):
            # if MultiPolygon, take the smallest convex Polygon containing all the points in the object
            poly = poly.convex_hull
        if (poly.geom_type == 'Polygon'):  # Ignore if still not a Polygon (could be a line or point)
            segmentation = np.array(poly.exterior.coords).ravel().tolist()
            if segmentation == []:
                continue
            polygons.append(poly)
            segmentations.append(segmentation)
    if not polygons == []:
        # Combine the polygons to calculate the bounding box and area
        multi_poly = MultiPolygon(polygons)
        x, y, max_x, max_y = multi_poly.bounds
        width = max_x - x
        height = max_y - y
        bbox = (x, y, width, height)
        area = multi_poly.area

        annotation = {
            'segmentation': segmentations,
            'iscrowd': is_crowd,
            'image_id': image_id,
            'category_id': category_id,
            'bbox': bbox,
            'area': area
        }
        return annotation


def mp_mask2coco(idx, create_sub_masks, create_sub_mask_annotation):
    is_crowd = 0
    mask_image = idx['value']
    key = idx['key'] + 1
    images = [{'file_name': idx['name'], 'width': idx['width'], 'height': idx['height'], 'id': key}]

    annotations = []
    sub_masks = create_sub_masks(mask_image)
    for color, sub_mask in sub_masks.items():
        category_id = color
        annotation = create_sub_mask_annotation(sub_mask, key, category_id, is_crowd)
        if annotation is not None:
            annotations.append(annotation)
    return annotations, images


def mp_image_add_id(item: dict):
    item['value']['id'] = item['key']
    return item['value']


def main(args: dict):
    category = args['category']
    directory = args['directory']
    mask_dir_path = Path(f'{args["path"]}/{category}/{directory}')

    mask_images = []
    for path in mask_dir_path.glob("**/*"):
        if path.suffix in ['.png']:
            mask_images += [Image.open(path)]


    mask_images_keys = list(range(len(mask_images)))
    mask_images_pairs = [{'key': k, 'value': v, 'name': Path(v.filename).name, 'width': v.size[0], 'height': v.size[1]} for k, v in dict(zip(mask_images_keys, mask_images)).items()]
    with Pool() as p:
        result = list(tqdm(p.imap(partial(
            mp_mask2coco, create_sub_masks=create_sub_masks, create_sub_mask_annotation=create_sub_mask_annotation
        ), mask_images_pairs), total=len(mask_images_pairs)))
    annotations, images = [j for i in result for j in i[0]], [j for i in result for j in i[1]]


    annotations_keys = list(range(len(annotations)))
    annotations_pairs = [{'key': k, 'value': v} for k, v in dict(zip(annotations_keys, annotations)).items()]
    with Pool() as p:
        annotations_pool = list(tqdm(p.imap(partial(
            mp_image_add_id
        ), annotations_pairs), total=len(annotations_pairs)))

    with open(Path(f'{args["path"]}/{category}/part_list.txt'), 'r') as f:
        category_labels = f.read().split('\n')
    category_labels = [i for i in category_labels if i != '']
    category_range = list(range(len(category_labels)))
    category_ids = dict(zip(category_range, category_labels))
    categories = []
    for k, v in category_ids.items():
        categories.append({"supercategory": category, 'id': k, 'name': v})

    output = {'images': images, 'annotations': annotations_pool, 'categories': categories}
    output_json = json.dumps(output, indent=4)
    output_path = Path(f'{args["output"]}/')
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path.joinpath(f'{category}_{directory}.json')
    with open(output_file, 'w') as f:
        f.write(output_json)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert dataset to coco from mask image')
    parser.add_argument('--path', '-p', type=str, default='/home/hpds/yucheng/dataset/ds_version', help='path')
    parser.add_argument('--category', '-c', type=str, default='aeroplane', help='category')
    parser.add_argument('--directory', '-d', type=str, default='seg', help='directory')
    parser.add_argument('--output', '-o', type=str, default='./output', help='output')
    args = parser.parse_args()

    main(vars(args))
