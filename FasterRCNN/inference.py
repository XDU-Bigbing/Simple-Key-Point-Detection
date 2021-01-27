# This script does only inference from the loaded model
import cv2
import matplotlib.pyplot as plt
import torch
import model
import os
from PIL import Image
import torchvision.transforms as T
import config

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

__all__ = [
    "load_model",
    "load_image_tensor",
    "get_prediction",
    "draw_box",
    "load_image_to_plot",
    "save_prediction",
    "get_folder_results",
]


def load_model():
    detector = model.create_model(num_classes=config.NUM_CLASSES)
    # print(detector)
    detector.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    # print(detector)
    detector.eval()
    detector.to(device)
    return detector


# Load the detector for inference


def load_image_tensor(image_path, device):
    image_tensor = T.ToTensor()(Image.open(image_path))
    input_images = [image_tensor.to(device)]
    return input_images


def get_prediction(detector, images):
    # We can do a batch prediction as well but right now I'm doing on single image
    # Batch prediction can improve time but let's keep it simple for now.
    with torch.no_grad():
        prediction = detector(images)
        return prediction


def draw_box(image, box, label_id, score):
    xtl = int(box[0])
    ytl = int(box[1])
    xbr = int(box[2])
    ybr = int(box[3])
    # Some hard coding for label
    if label_id == 1:
        label = "yes"
        cv2.rectangle(image, (xtl, ytl), (xbr, ybr), color=(0, 255, 0))
    elif label_id == 2:
        label = "no"
        cv2.rectangle(image, (xtl, ytl), (xbr, ybr), color=(0, 0, 255))
    elif label_id == 3:
        label = "invisible"
        cv2.rectangle(image, (xtl, ytl), (xbr, ybr), color=(0, 0, 255))
    elif label_id == 4:
        label = "wrong"
        cv2.rectangle(image, (xtl, ytl), (xbr, ybr), color=(0, 0, 255))

    print("label = {}".format(label))
    cv2.putText(
        image, label, (xtl, ytl), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2
    )
    # cv2.putText(image, label, (xbr, ybr), cv2.FONT_HERSHEY_SIMPLEX,  0.5, (36,255,12), 2)


def load_image_to_plot(image_dir):
    image = cv2.imread(image_dir, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def save_prediction(prediction, image_name, image):
    for pred in prediction:
        boxes = pred["boxes"].data.cpu().numpy()
        labels = pred["labels"].data.cpu().numpy()
        scores = pred["scores"].data.cpu().numpy()

    for i in range(len(labels)):
        if scores[i] > config.DETECTION_THRESHOLD:
            box_draw = boxes[i]
            label_draw = labels[i]
            score = scores[i]
            print(score)
            print(box_draw)
            print(label_draw)
            draw_box(image, box_draw, label_draw, score)

    # plt.imshow(image)
    # plt.show()

    # image_name = config.OUTPUT_PATH + image_name
    cv2.imwrite(image_name, image)


def get_folder_results(detector, image_dir, device):
    for image in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image)
        input_images = load_image_tensor(image_path, device)
        prediction = get_prediction(detector, input_images)
        image_loaded = load_image_to_plot(image_path)
        save_path = os.path.join(config.SAVE_DIR, image)
        save_prediction(prediction, save_path, image_loaded)


if __name__ == "__main__":
    detector = load_model()
    print("---------- Model succesfully loaded -------- ")
    # print(detector)

    input_images = load_image_tensor(config.PREDICT_IMAGE, device)
    prediction = get_prediction(detector, input_images)
    # print(prediction)
    image = load_image_to_plot(config.PREDICT_IMAGE)
    # save_prediction(prediction, config.SAVE_IMAGE, image)
    get_folder_results(detector, config.IMAGE_DIR, device)
    # print(prediction)
