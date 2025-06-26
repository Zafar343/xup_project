import cv2
import os
import numpy as np
import glob
import tqdm
import argparse
from loguru import logger

def annotate(img, label):
    height, width, _ = img.shape
    for l in label:
        c, x, y, w, h = l

        x1 = (x * width) - ((w * width) / 2)
        y1 = (y * height) - ((h * height) / 2)
        x2 = (x * width) + (w * width/2)
        y2 = (y * height) + (h * height/2)

        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(img, f"{c}", (int(x1+3), int(y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return img

def main(path, save=False):

    logPath = "logs"
    if not os.path.exists(logPath):
        os.makedirs(logPath, exist_ok=True)

    logger.add(os.path.join(logPath, "visulalize.log"), rotation="1 day", level="TRACE",
                                retention="60 days", compression="zip",
                                enqueue=True, backtrace=True, diagnose=True, colorize=False)

    imgPaths = glob.glob(os.path.join(path, "*.jpg"))
    noLabels = 0
    for imgPath in tqdm.tqdm(imgPaths):
        labelPath = imgPath.replace("images", "labels").replace(".jpg", ".txt")
        if not os.path.exists(labelPath):
            noLabels += 1
            continue
        
        label = np.loadtxt(labelPath, dtype=np.float32)
        
        if label.ndim == 1 and len(label)>0:
            label = label[np.newaxis, :]
        
        image = cv2.imread(imgPath)
        
        if len(label) > 0:
            image = annotate(image, label)
        else:
            noLabels += 1

        imgName = imgPath.split("/")[-1]
        if save:
            cv2.imwrite(os.path.join('analysis', imgName), image)

        cv2.imshow(imgName, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    logger.info(f"Number of images with no labels: {noLabels}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=str, default="/data/Datasets/Golf_project/dataset_split/train/images", help="Path to the dataset directory")
    args = parser.parse_args()
    
    main(args.paths)