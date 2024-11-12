import tiktoken
import torch
import nltk
from nltk.tokenize import sent_tokenize
import math 

def counting_bigram(tokens):
    index = max(tokens) + 1
    N = torch.zeros((index, index), dtype=torch.int32)

    for i, j in zip(tokens, tokens[1:]):
        N[i,j] += 1

    return N

def tokenizador_tiktoken(tokenizer, texto):
    frases = sent_tokenize(texto, language='portuguese')
    frases = [f"{frase}<|endoftext|>" for frase in frases]

    texto = "".join(frases)

    # Tokenizar o texto em uma lista de IDs de tokens
    tokens = tokenizer.encode(texto, allowed_special={'<|endoftext|>'})

    # Definir proporção do conjunto de dados de treino e teste (80% treino, 20% teste)
    train_size = int(0.8 * len(tokens))

    train_tokens = tokens[:train_size]
    test_tokens = tokens[train_size:]

    return train_tokens, test_tokens

def probabilidade_condicionais(N):
    P = (N).float()
    P /= P.sum(1, keepdims=True)
    g = torch.Generator().manual_seed(2147483647)
    return P, g

def proximo_token(P, g, ix, tokenizer):
    out = []
    while True:
        p = P[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        if ix == 50256:
            break
        out.append(tokenizer.decode([ix]))
    return out, ix

def perplexidade(train_tokens, N):
    n_total = len(train_tokens)
    prob_produto = 1.0
    for i in range (1,n_total):
        w_i = train_tokens[i]
        w_i_1 = train_tokens[i-1]
        prob = N[w_i,w_i_1]

        if prob == 0:
            prob = 1e-10
        prob_produto *= 1/prob

    perplexidade = prob_produto**(1/n_total)        
    print(perplexidade)

def gerar_texto(train_tokens, P, g, tokenizer):
    # Testar a previsão
    ix = train_tokens[0]
    for i in range(20): 
        out, ix = proximo_token(P, g, ix, tokenizer)
        print({f'{i+1}: {"".join(out)}'})