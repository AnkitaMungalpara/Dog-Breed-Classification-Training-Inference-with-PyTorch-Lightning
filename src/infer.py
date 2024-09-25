import argparse
from models.dog_classifier import DogClassifier
import torch
import os
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

def inference(model, image_path):

    # load image
    img = Image.open(image_path).convert("RGB")

    # define transformations
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # now, apply these transformations to an image
    img_tensor = transform(img).unsqueeze(0)
    img_tensor = img_tensor.to(model.device)

    # set the model in evaluation mode
    model.eval()

    # perform inference
    with torch.no_grad():
        output = model(img_tensor)
        probabilty = F.softmax(output, dim=1)
        predicted = torch.argmax(probabilty, dim=1).item()

    # map the predicted class to label
    class_labels = [
        "Beagle",
        "Boxer",
        "Bulldog",
        "Dachshund",
        "German_Shepherd",
        "Golden_Retriever",
        "Labrador_Retriever",
        "Poodle",
        "Rottweiler",
        "Yorkshire_Terrier",
    ]
    predicted_label = class_labels[predicted]
    confidence = probabilty[0][predicted].item()

    return img, predicted_label, confidence


def save_prediction(img, actaul_label, predicted_label, confidence, output_path):
    plt.figure(figsize=(9, 9))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Actual: {actaul_label} | Predicted: {predicted_label} | (Confidence: {confidence:.2f})")
    plt.savefig(output_path)
    plt.show()
    plt.close()


def main(args):

    # load model
    model = DogClassifier.load_from_checkpoint(args.ckpt_path)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # create a directory for storing predictions if not exists
    os.makedirs(args.output_folder, exist_ok=True)

    # get list of files
    files = [
        f
        for f in os.listdir(args.input_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    for file in random.sample(files, min(10, len(files))):
        image_path = os.path.join(args.input_folder, file)
        img, predicted_label, confidence = inference(model=model, image_path=image_path)

        # saving the prediction image
        output_image_path = os.path.join(
            args.output_folder, f"{os.path.splitext(file)[0]}_prediction.png"
        )

        actaul_label = image_path.split("/")[-1].split('_')[0]
        save_prediction(img, actaul_label, predicted_label, confidence, output_image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform inference on images")

    parser.add_argument(
        "--input_folder",
        type=str,
        default="data/test",
        help="Path to the directory containing input images",
    )

    parser.add_argument(
        "--output_folder",
        type=str,
        default="predictions",
        help="Path to the directory contaning predictions",
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="checkpoints/dog_breed_classifier_model.ckpt",
        help="path to the model checkpoint",
    )

    args = parser.parse_args()
    main(args)
