from pytorch_lightning import LightningModule
import torch
from torchmetrics import MaxMetric, MeanMetric
from typing import Any, List
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from PIL import Image
from torch import nn

class VitGpt2(LightningModule):
    '''
    Load pre-traine Vit-GPT2 model from (https://huggingface.co/nlpconnect/vit-gpt2-image-captioning) 
    - fine-tune on other datasets
    - VisionEncoderDecoderModel: uses visuon model as encoder, and language model as decoder
        - implements cross attention to pay attention to visual features and generate text outputs
        - training: https://huggingface.co/docs/transformers/model_doc/vision-encoder-decoder
    '''
    def __init__(self, optimizer, scheduler, data_dir: str, max_length = 16, num_beams = 4):
        super().__init__()
    
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

        self.image_dir = data_dir + "images/"

        self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
    
    def forward(self, input):
        '''
        Load image path and text caption taget
        - reference: https://huggingface.co/nlpconnect/vit-gpt2-image-captioning
        '''
        image_names = input["images"]
        caption_target = input["caption"]
        logits = 0
        loss = 0
        images = []

        for name in image_names:
            i_image = Image.open(self.image_dir + name)
            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")

            images.append(i_image)

        import pdb
        pdb.set_trace()

        pixel_values = self.feature_extractor(images=images, return_tensors="pt").pixel_values # (bs, num_channels, w, h)
        pixel_values = pixel_values.to(self.device)


        # generate output logits
        loss = self.model(pixel_values=pixel_values, labels=caption_target).loss

        # generate caption
        output_ids = self.model.generate(pixel_values, **self.gen_kwargs)
        output_caption = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        # caption = [pred.strip() for pred in preds]

        return {"loss": loss, "output_caption": output_caption}
    
    def training_step(self, batch: Any, batch_idx: int):

        import pdb
        pdb.set_trace()

        loss, preds, targets = self(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}


    def validation_step(self, batch: Any, batch_idx: int):

        import pdb
        pdb.set_trace()

        loss, preds, targets = self(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}
    
    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):

        import pdb
        pdb.set_trace()

        loss, preds, targets = self(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}