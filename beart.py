PATH = './sample_data/'
dados_treino = 'crepusculoDosIdolos.txt'

from tokenizers import ByteLevelBPETokenizer

# ---------------------------------- Treino do tokenizer ----------------------------------

# Inicializando o tokenizer
tokenizer = ByteLevelBPETokenizer()

# Aqui estamos usando o train para treinar nosso tokenizer em cima dos nossos dados_treino
tokenizer.train(files=[PATH+dados_treino], vocab_size=52_000, min_frequency=2, special_tokens=[
    "<s>", # Começo de linha
    "<pad>",  # Esse pad é o preenchimento de valores vazios, para quando ele não preencher o vetor completo, preencher os espaços vazios e evitar problemas.
    "</s>", # Final de frase
    "<unk>", # Caracter desconhecido
    "<mask>", # Local para colocar o caracter para fazer a predição
])

# Realizando alguns testes
encodedString = tokenizer.encode("Hoje é um novo dia").ids
decodedString = tokenizer.decode([44, 83, 570, 306, 300, 1714, 556, 5])

print("ida: ")
print(encodedString)
print("volta: ")
print(decodedString)

# Salvando meu tokenizer em uma pasta
tokenizer.save_model(PATH+'RAW_MODEL')
# Ao analisar o arquivo vocab.json, vemos que ali está criado nosso vocabulario, nosso dicionário. Ali está todo o mapeamento dos nossos tokens.

# Já o merges.txt são os pares que ele utilizou para treinar nosso tokenizer

# ---------------------------------- Construindo o tokenizer ----------------------------------

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