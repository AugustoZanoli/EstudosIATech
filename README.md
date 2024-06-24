------/Primeiro passo/------

Primeiro passo feito nesse projeto foi utilizar o comando "!pip install transformers[torch]", para instalarmos
no nosso projeto as bibliotecas e pacotes necessários para termos acesso aos transformers

------/Segundo passo/------

Instalar o livro "Crepusculo dos idolos" de nietzsche, a fim de usa-lo como corpus. Esse é um corpus bem robusto,
porém não é tão gigante ao ponto de ser um problema para o computador ou para os treinos.

"!wget -O ./sample_data/crepusculoDosIdolos.txt https://raw.githubusercontent.com/mfmarlonferrari/NietzscheGPT/main/crepusculoDosIdolos.txt"

------/BytelevelBPETokenizer/------

Faz uma recursão para analisar as frases.

![image](https://github.com/AugustoZanoli/EstudosIATech/assets/143662315/58e0d8ae-76fe-4b36-b90c-0f6af0de9996)

------/Inicializando e treinando o tokenizer/------

"s", # Começo de linha

"pad",  # Esse pad é o preenchimento de valores vazios, para quando ele não preencher o vetor completo, preencher os espaços vazios e evitar problemas.

"/s", # Final de frase

"unk", # Caracter desconhecido

"mask", # Local para colocar o caracter para fazer a predição

