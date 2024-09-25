import argparse
from models.dog_classifier import DogClassifier
import torch
from datamodules.dogbreed import DogImageDataModule
import lightning as L


def main(args):

    # data module
    data_module = DogImageDataModule(num_workers=2, batch_size=16)

    # set up the data module for validation data
    data_module.setup(stage="fit")

    # validation datset
    # val_dataset = data_module.val_dataset

    # data loader
    val_data_loader = data_module.val_dataloader()

    # load model
    model = DogClassifier.load_from_checkpoint(args.ckpt_path)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Trainer
    trainer = L.Trainer(
        # limit_train_batches=0.05,
        # limit_test_batches=0.05,
        max_epochs=1,
        log_every_n_steps=10,
        accelerator="auto",
    )

    # test the module
    results = trainer.test(model=model, datamodule=data_module)

    print("Validation is completed!!!")
    print(f"validation results: {results}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform evaluation on images")

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="checkpoints/dog_breed_classifier_model.ckpt",
        help="path to the model checkpoint",
    )

    args = parser.parse_args()
    main(args)
