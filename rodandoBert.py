PATH = './sample_data/'
dados_treino = 'crepusculoDosIdolos.txt'

from transformers import pipeline

fill_mask = pipeline(
    "fill_mask",
    model=PATH+'RAW_MODEL',
    tokenizer=PATH+'RAW_MODEL',
)