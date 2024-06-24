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