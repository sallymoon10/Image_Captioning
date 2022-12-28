from utils import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=["README.md"],
    pythonpath=True,
    dotenv=True,
)

import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu

from utils import pylogger, utils
from typing import List, Optional, Tuple
import hydra
from PIL import Image
from rouge import Rouge


log = pylogger.get_pylogger(__name__)


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """
    Evaluate fine-tuned model from checkpoint
    """
    assert cfg.ckpt_path
    import pdb
    pdb.set_trace()

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    add_configs_from_datamodule(cfg = cfg, datamodule=datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    best_model = model.load_from_checkpoint(cfg.ckpt_path)
    print_eval_metrics(best_model=best_model, datamodule=datamodule)


def add_configs_from_datamodule(cfg: DictConfig, datamodule):
    cfg.model.data_dir = datamodule.data_dir

def print_eval_metrics(best_model, datamodule, image_col = 'image', caption_col = 'caption'):
    import pdb
    pdb.set_trace()

    test_input = {"images": list(datamodule.df_test[image_col]), "caption": list(datamodule.df_test[caption_col])}
    gen_captions = list(best_model(test_input)["output_caption"])
    truth_captions = list(datamodule.df_test[caption_col])

    print('--------------------------------------------------------------------\n')
    print('EVALUATION ON TEST SET')
    bleu = get_bleu_score(gen_captions=gen_captions, truth_captions=truth_captions)
    rouge = get_rouge_score(gen_captions=gen_captions, truth_captions=truth_captions)
    print(f'BLEU score: \n {bleu}')
    print(f'ROUGE score: \n {rouge}')

    visualize_results_on_sample(best_model=best_model, datamodule=datamodule)

def get_bleu_score(gen_captions: list, truth_captions: list):
    ''' 
    Calculate pair wise bleu score 
    - Bleu score is a precision-focused metric based on calculating n-gram overlap between generated and truth sentences
        - Precision - how much n-grams in gen_captions appear in truth_caption (i.e. of all n-grams in generated caption, how many appear in truth caption) (how many generated tokens are correct)
        - n-gram = set of 'n' consecutive words in a sentence 
        - has brevity penalty, meaning it penalizes when generated sentence is significantly shorter than truth sentence 
    - Reference: https://towardsdatascience.com/how-to-evaluate-text-generation-models-metrics-for-automatic-evaluation-of-nlp-models-e1c251b04ec1
    
    Inputs:
        gen_captions : a list of reference sentences 
        truth_captions : a list of candidate(generated) sentences
    Returns:
        bleu score(float)
    '''
    truth_bleu = []
    gen_bleu = []
    for caption in gen_captions:
        gen_bleu.append(caption.split())

    for caption in truth_captions:
        truth_bleu.append([caption.split()])

    cc = SmoothingFunction()
    score_bleu = corpus_bleu(truth_bleu, gen_bleu, weights=(0, 1, 0, 0), smoothing_function=cc.method4)
    return score_bleu

def get_rouge_score(gen_captions: list, truth_captions: list):
    '''
    Calculate rouge score 
    - Rouge score is a set of recall-oriented metrics 
        - Recall - how much n-grams in gen_captions appear in truth_caption (i.e. of all n-grams in the truth caption, how many were generated) (how many true n-grams were recalled)
        - Rouge-N: measures number of matching n-grams between generated and truth sentence
        - Rouge-L: measures longest common subsequence 
    - Reference: https://towardsdatascience.com/the-ultimate-performance-metric-in-nlp-111df6c64460
    '''
    rouge = Rouge()
    rouge_scores = rouge.get_scores(gen_captions, truth_captions, avg=True)

    return rouge_scores


def visualize_results_on_sample(best_model, datamodule, image_col = 'image', caption_col = 'caption'):
    # sample from test set 
    sample_test = datamodule.df_test.sample(n=10)
    images = list(sample_test[image_col])
    truth_captions = list(sample_test[caption_col])
    pred_captions = best_model(images)["output_caption"]
    
    print('VISUALIZE MODEL OUTPUTS ON SAMPLE SET')
    for i, name in enumerate(images):
        i_image = Image.open(datamodule.data_dir + 'images/' + name)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")
        
        print('\n\n--------------------------------------------------------------------\n')
        i_image.show()
        print(f'Ground truth caption: {truth_captions[i]}\n')
        print(f'Predited caption: {pred_captions}')


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)

if __name__ == "__main__":
    main()