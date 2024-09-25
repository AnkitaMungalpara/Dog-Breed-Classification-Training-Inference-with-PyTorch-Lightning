import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from datamodules.dogbreed import DogImageDataModule
from models.dog_classifier import DogClassifier
from utils.utils import task_wrapper
from utils.pylogger import get_pylogger
import argparse
import os

log = get_pylogger(__name__)


def train(args):

    # 1. data module
    data_module = DogImageDataModule(data_dir="data", batch_size=32)

    # 2. set up model
    if os.path.exists(args.ckpt_path):
        model = DogClassifier.load_from_checkpoint(args.ckpt_path)
    else:
        model = DogClassifier(lr=1e-3)

    # 3. define logger
    logger = TensorBoardLogger(save_dir="logs", name="catdog_classifier")

    # 4. set up callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath="checkpoints/",
        filename="dog_breed_classifier_model",
        save_top_k=1,
        mode="min",
    )

    # 5. Define Trainer
    trainer = L.Trainer(
        max_epochs=5,
        callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=10,
        accelerator="auto",
    )

    config = {"data": vars(data_module), "model": vars(model), "trainer": vars(trainer)}

    # train the model
    if not os.path.exists(args.ckpt_path):
        log.info('Started model training as no checkpoint found.')
        trainer.fit(model=model, datamodule=data_module)

    # test the module
    trainer.test(model=model, datamodule=data_module)

    log.info("finishing up")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform evaluation on images")

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="checkpoints/dog_breed_classifier_model.ckpt",
        help="path to the model checkpoint",
    )

    args = parser.parse_args()
    train(args)
