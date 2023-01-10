from pytorch_lightning import LightningModule
import torch
from torchmetrics import MinMetric, MeanMetric
from typing import Any, List, Optional
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from PIL import Image
from torch import nn
from utils.utils import freeze_model_and_make_eval

class VitGpt2(LightningModule):
    '''
    - Load pre-trained Vit-GPT2 model from: https://huggingface.co/nlpconnect/vit-gpt2-image-captioning
    - Fine-tune on other datasets (i.e. Flickr, Medicat)
    - VisionEncoderDecoderModel: uses vision model as encoder, and language model as decoder
        - implements cross attention to pay attention to visual features and generate text outputs
        - training notes can be found here: https://huggingface.co/docs/transformers/model_doc/vision-encoder-decoder
    '''
    def __init__(self, optimizer: torch.optim, scheduler: torch.optim, data_dir: str, max_length: Optional[int] = 16, num_beams: Optional[int]  = 4, num_layers_to_train: Optional[int]  = 1):
        '''
        Inputs:
        - optimizer: optimizer to train model, set up in configs/model/vit_gpt2.yaml
        - scheduler: schedules the learning rate throughout training, set up in configs/model/vit_gpt2.yaml
        - data_dir: directory containing data (images and corresponding captions)
        - max_length: max_length of generated caption
        - num_beams: number of beams to generate caption
        - num_layers_to_train: number of layers from the last layer of encoder and decoder to unfreeze and train during fine-tuning
        '''
        super().__init__()
        self.save_hyperparameters(logger=False)  # allows init params to be stored in ckpt
    
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
        self.num_layers_to_train = num_layers_to_train

        self.image_dir = data_dir + "images/"
        self.set_up_model_for_training()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()

    def set_up_model_for_training(self):
        '''Only unfreeze last layer of encoder and decoder. To fine-tune small number of layers under low resource setting.'''
        self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        freeze_model_and_make_eval(self.model)
        self.unfreeze_last_layers()

    def unfreeze_last_layers(self):
        '''
        Unfreeze layer layer of encoder and decoder
        - Manually identified last layers to unfreeze for demonstration purposes
        '''
        encoder_last_layer = ['model.encoder.encoder.layer.11.attention.attention.query.bias', 'model.encoder.encoder.layer.11.attention.attention.key.weight', 'model.encoder.encoder.layer.11.attention.attention.key.bias', 'model.encoder.encoder.layer.11.attention.attention.value.weight', 'model.encoder.encoder.layer.11.attention.attention.value.bias', 'model.encoder.encoder.layer.11.attention.output.dense.weight', 'model.encoder.encoder.layer.11.attention.output.dense.bias', 'model.encoder.encoder.layer.11.intermediate.dense.weight', 'model.encoder.encoder.layer.11.intermediate.dense.bias', 'model.encoder.encoder.layer.11.output.dense.weight', 'model.encoder.encoder.layer.11.output.dense.bias', 'model.encoder.encoder.layer.11.layernorm_before.weight', 'model.encoder.encoder.layer.11.layernorm_before.bias', 'model.encoder.encoder.layer.11.layernorm_after.weight', 'model.encoder.encoder.layer.11.layernorm_after.bias']
        decoder_last_layer = ['model.decoder.transformer.h.11.ln_1.weight', 'model.decoder.transformer.h.11.ln_1.bias', 'model.decoder.transformer.h.11.attn.c_attn.weight', 'model.decoder.transformer.h.11.attn.c_attn.bias', 'model.decoder.transformer.h.11.attn.c_proj.weight', 'model.decoder.transformer.h.11.attn.c_proj.bias', 'model.decoder.transformer.h.11.ln_2.weight', 'model.decoder.transformer.h.11.ln_2.bias', 'model.decoder.transformer.h.11.crossattention.c_attn.weight', 'model.decoder.transformer.h.11.crossattention.c_attn.bias', 'model.decoder.transformer.h.11.crossattention.q_attn.weight', 'model.decoder.transformer.h.11.crossattention.q_attn.bias', 'model.decoder.transformer.h.11.crossattention.c_proj.weight', 'model.decoder.transformer.h.11.crossattention.c_proj.bias', 'model.decoder.transformer.h.11.ln_cross_attn.weight', 'model.decoder.transformer.h.11.ln_cross_attn.bias', 'model.decoder.transformer.h.11.mlp.c_fc.weight', 'model.decoder.transformer.h.11.mlp.c_fc.bias', 'model.decoder.transformer.h.11.mlp.c_proj.weight', 'model.decoder.transformer.h.11.mlp.c_proj.bias']

        for name, param in dict(self.named_parameters()).items():
            if name in encoder_last_layer or name in decoder_last_layer:
                param.requires_grad = True
        
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
        images = []

        for name in image_names:
            i_image = Image.open(self.image_dir + name)
            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")

            images.append(i_image)

        pixel_values = self.feature_extractor(images=images, return_tensors="pt").pixel_values # (bs, num_channels, w, h)
        pixel_values = pixel_values.to(self.device)

        caption_target_ids = self.tokenizer(caption_target, return_tensors="pt", padding=True, truncation=True).input_ids
        caption_target_ids = caption_target_ids.to(self.device)

        # generate output logits
        output = self.model(pixel_values=pixel_values, labels=caption_target_ids)
        loss = output.loss
        logits = output.logits

        # generate caption
        output_ids = self.model.generate(pixel_values, **self.gen_kwargs)
        output_caption = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        return {"loss": loss, "output_caption": output_caption, "logits": logits}
    
    def on_after_backward(self):
        global_step = self.global_step
        for name, param in self.named_parameters():
            if param.requires_grad and param.grad is not None:
                abs_mean_grad = abs(param.grad).mean()
                self.logger.experiment.add_histogram(f"{name}_grad", param.grad, global_step)
                self.logger.experiment.add_scalar(f"{name}_abs_mean_grad", abs_mean_grad, global_step)

    def training_step(self, batch: Any, batch_idx: int):
        output = self(batch)

        self.train_loss(output["loss"])
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_logits(logits=output["logits"], name="train/logits")
        return output

    def training_epoch_end(self, outputs: List[Any]):
        self.log_params_as_histogram()

    def validation_step(self, batch: Any, batch_idx: int):
        output = self(batch)
        # update and log metrics
        self.val_loss(output["loss"])
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_logits(logits=output["logits"], name="val/logits")
        return output
    
    def validation_epoch_end(self, outputs: List[Any]):
        loss = self.val_loss.compute() 
        self.val_loss_best(loss) 
        self.log("val/loss_best", self.val_loss_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        output = self(batch)

        self.test_loss(output["loss"])
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_logits(logits=output["logits"], name="test/logits")
        return output
    
    def log_params_as_histogram(self):
        '''Save weight histogram to tensorboard'''
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)
    
    def log_logits(self, logits: torch.Tensor, name: str):
        '''Save logit histogram to tensorboard'''
        self.logger.experiment.add_histogram(name, logits, self.current_epoch)