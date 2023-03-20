from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd
from os import listdir
from sklearn.model_selection import train_test_split

class CartoonDatamodule(LightningDataModule):
    """
    Datamodule for NY cartoon dataset
    - dataset: https://github.com/nextml/caption-contest-data
    """

    def __init__(
        self,
        data_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        batch_size: int = 8,
        num_workers: int = 0,
        pin_memory: bool = False,
        num_samples = None,
        image_col: str = 'image',
        caption_col:str = 'caption'

    ):
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.data_dir = data_dir
        self.image_col = image_col
        self.caption_col = caption_col
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.num_samples = num_samples

        self.df_train, self.df_val, self.df_test = self.prepare_data()

    def read_caption(self, image: str, summary_file = '_LilUCB.csv'):
        try:
            caption_df = pd.read_csv(image + summary_file)
            return caption_df['caption'].iloc[0]

        except Exception as e:
            return None


    def prepare_data(self, random_state = 100):
        breakpoint()
        self.df = pd.DataFrame()
        caption_data = listdir(self.data_dir + 'summaries')
        self.df[self.image_col] = listdir(self.data_dir + 'images')
        self.df[self.caption_col] = self.df[self.image_col].apply(lambda image: self.read_caption(image))

        
        self.df = pd.read_csv(self.data_dir + "captions.txt")

    
        # filter for those with valid text
        self.df = self.df.loc[self.df[self.caption_col].notnull()]

        if self.num_samples:
            self.df = self.df.iloc[:self.num_samples]

        images = self.df[self.image_col].values

        # add train, val, test split
        train_images, rest_images = train_test_split(images, train_size = self.train_ratio, shuffle = False, random_state = random_state)
        val_images, test_images = train_test_split(rest_images, test_size = self.test_ratio/ (1 - self.train_ratio), shuffle = False, random_state = random_state)

        df_train = self.df.loc[self.df[self.image_col].isin(train_images)]
        df_val = self.df.loc[self.df[self.image_col].isin(val_images)]
        df_test = self.df.loc[self.df[self.image_col].isin(test_images)]

        return df_train, df_val, df_test

    def train_dataloader(self):
        images = self.df_train[self.image_col].values
        captions = self.df_train[self.caption_col].values

        image_data_loader = DataLoader(
            dataset=images,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

        caption_data_loader = DataLoader(
            dataset=captions,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

        return CombinedLoader({"images": image_data_loader, "caption": caption_data_loader})


    def val_dataloader(self):
        images = self.df_val[self.image_col].values
        captions = self.df_val[self.caption_col].values

        image_data_loader = DataLoader(
            dataset=images,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

        caption_data_loader = DataLoader(
            dataset=captions,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

        return CombinedLoader({"images": image_data_loader, "caption": caption_data_loader})

    def test_dataloader(self):
        images = self.df_test[self.image_col].values
        captions = self.df_test[self.caption_col].values

        image_data_loader = DataLoader(
            dataset=images,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

        caption_data_loader = DataLoader(
            dataset=captions,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

        return CombinedLoader({"images": image_data_loader, "caption": caption_data_loader})
