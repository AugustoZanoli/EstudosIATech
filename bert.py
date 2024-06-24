PATH = './sample_data/'
dados_treino = 'crepusculoDosIdolos.txt'

# ---------------------------------- Instanciando o tokenizer ----------------------------------

# Cria um tokenizer baseado no algoritmo BPE

from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing # Modelo bi-direcional

# Instanciando meu modelo

#tokenizer = ByteLevelBPETokenizer(
#    PATH+'RAW_MODEL'+"vocab.json",
#    PATH+'RAW_MODEL'+"merges.txt",
#)
#tokenizer._tokenizer.post_processor = BertProcessing(
#    ("</s>", tokenizer.token_to_id("</s>")),
#    ("<s>", tokenizer.token_to_id("<s>")),
#)
#tokenizer.enable_truncation(max_length=512)

# Ao invés de usar o BERT, irei usar RobertaTokenizer

from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained(PATH+'RAW_MODEL', max_length=512)

# ---------------------------------- Criando o transformer ----------------------------------

from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=512,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

from transformers import RobertaForMaskedLM
model= RobertaForMaskedLM(config=config)

# print(model.num_parameters()) ver a quantidade de parametros existentes na nossa rede neural

# ---------------------------------- Criando o DataSet tokenizado ----------------------------------

# Forma simples de se carregar um arquivo bruto como Dataset

from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=PATH+dados_treino,
    block_size=128,
)

# Verificando

print(dataset.examples[:2])

print(tokenizer.decode(dataset.examples[7]['input_ids']))