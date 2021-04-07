from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import random
import os
import cv2


def filterDataset(folder, classes=None, mode='train'):
    print('filterDataset')
    # initialize COCO api for instance annotations
    annFile = '{}/coco.json'.format(folder)
    print(annFile)
    coco = COCO(annFile)

    images = []
    if classes is not None:
        # iterate for each individual class in the list
        for className in classes:
            # get all images containing given categories
            catIds = coco.getCatIds(catNms=className)
            imgIds = coco.getImgIds(catIds=catIds)
            images += coco.loadImgs(imgIds)

    else:
        imgIds = coco.getImgIds()
        images = coco.loadImgs(imgIds)

    # Now, filter out the repeated images
    unique_images = []
    for i in range(len(images)):
        if images[i] not in unique_images:
            unique_images.append(images[i])

    random.shuffle(unique_images)
    dataset_size = len(unique_images)

    return unique_images, dataset_size, coco


def getClassName(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id'] == classID:
            return cats[i]['name']
    return None


def getImage(imageObj, img_folder, input_image_size):
    # Read and normalize an image
    train_img = io.imread(img_folder + '/' + imageObj['file_name']) / 255.0
    # Resize
    train_img = cv2.resize(train_img, input_image_size)
    if len(train_img.shape) == 3 and train_img.shape[2] == 3:  # If it is a RGB 3 channel image
        return train_img
    else:  # To handle a black and white image, increase dimensions to 3
        stacked_img = np.stack((train_img,) * 3, axis=-1)
        return stacked_img


def getNormalMask(imageObj, classes, coco, catIds, input_image_size):
    annIds = coco.getAnnIds(imageObj['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    cats = coco.loadCats(catIds)
    train_mask = np.zeros(input_image_size)
    for a in range(len(anns)):
        className = getClassName(anns[a]['category_id'], cats)
        pixel_value = classes.index(className) + 1
        new_mask = cv2.resize(coco.annToMask(anns[a]) * pixel_value, input_image_size)
        train_mask = np.maximum(new_mask, train_mask)

    # Add extra dimension for parity with train_img size [X * X * 3]
    train_mask = train_mask.reshape(input_image_size[0], input_image_size[1], 1)
    return train_mask


def getBinaryMask(imageObj, coco, catIds, input_image_size):
    annIds = coco.getAnnIds(imageObj['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    train_mask = np.zeros(input_image_size)
    for a in range(len(anns)):
        new_mask = cv2.resize(coco.annToMask(anns[a]), input_image_size)

        # Threshold because resizing may cause extraneous values
        new_mask[new_mask >= 0.5] = 1
        new_mask[new_mask < 0.5] = 0

        train_mask = np.maximum(new_mask, train_mask)

    # Add extra dimension for parity with train_img size [X * X * 3]
    train_mask = train_mask.reshape(input_image_size[0], input_image_size[1], 1)
    return train_mask


def dataGeneratorCoco(images, classes, folder, input_image_size=(224, 224), batch_size=4, mode='train',
                      mask_type='binary'):
    print("###############################################")
    print("#          Start dataGeneratorCoco            #")
    print("###############################################")

    annFile = '{}/coco.json'.format(folder)
    print("Annotation file: {}".format(annFile))

    # initialize the COCO api for instance annotations
    coco = COCO(annFile)

    # display COCO categories and supercategories
    catIDs = coco.getCatIds()
    cats = coco.loadCats(catIDs)

    nms = [cat['name'] for cat in cats]
    print(len(nms), 'COCO categories: \n{}\n'.format(' '.join(nms)))

    nms = set([cat['supercategory'] for cat in cats])
    print(len(nms), 'COCO supercategories: \n{}'.format(' '.join(nms)))

    img_folder = '{}/{}'.format(folder, mode)
    print("Source images folder: {}".format(img_folder))

    dataset_size = len(images)
    print("Dataset size: {}".format(dataset_size))

    catIds = coco.getCatIds(catNms=classes)
    print("Categories IDs: {}".format(catIds))

    c = 0
    while True:
        # print(c)
        img = np.zeros((batch_size, input_image_size[0], input_image_size[1], 3)).astype('float')
        mask = np.zeros((batch_size, input_image_size[0], input_image_size[1], 1)).astype('float')

        for i in range(c, c + batch_size):  # initially from 0 to batch_size, when c = 0
            imageObj = images[i]

            print("# Start processing image: {}".format(imageObj['file_name']))

            ### Retrieve Image ###
            train_img = getImage(imageObj, img_folder, input_image_size)

            ### Create Mask ###
            if mask_type == "binary":
                train_mask = getBinaryMask(imageObj, coco, catIds, input_image_size)

            elif mask_type == "normal":
                train_mask = getNormalMask(imageObj, classes, coco, catIds, input_image_size)

                # Add to respective batch sized arrays
            img[i - c] = train_img
            mask[i - c] = train_mask

            # train_mask_save = transforms.ConvertImageDtype(dtype=torch.uint8) (train_mask)
            png_image_path = os.path.join('{}/{}'.format(folder, "PNGImages"),
                                          imageObj['file_name'].split(".")[0] + ".png")
            print(" - Saving image: {}".format(png_image_path))
            io.imsave(png_image_path, train_img)

            png_image_masked_path = os.path.join('{}/{}'.format(folder, "masks"),
                                                 imageObj['file_name'].split(".")[0] + "_mask.png")
            print(" - Saving masked image: {}".format(png_image_masked_path))
            io.imsave(png_image_masked_path, train_mask)

        c += batch_size
        if c + batch_size >= dataset_size:
            c = 0
            random.shuffle(images)
        return img, mask


folder = './dataset'
classes = ['sail']
mode = 'originals'

batch_size = 1
input_image_size = (800, 800)
mask_type = 'normal'

images, dataset_size, coco = filterDataset(folder, classes, mode)

print("Dataset Size: {}".format(dataset_size))
print(images)

dataGeneratorCoco(images, classes, folder, input_image_size, batch_size, mode, mask_type)
