import os
import json
import sys

def get_stats(ids):
     """
     É fornecido uma lista de valores inteiros que será
     percorida para encontrar os pares mais comuns desta
     lista. Para facilitar, é utilizado um dicionário.
     As tuplas (que são os pares) são as chaves do dicionário.

     ids são identificadores númericos.
     """
     pairs = {}
     for tokens in zip(ids, ids[1:]):
          if tokens in pairs:
               pairs[tokens] += 1
          else:
               pairs[tokens] = 1

     return pairs; 

def merge(pair, ids, swap_id):
     """
     Substituir o par de maior ocorrência por um novo id
     """
     new_ids = []
     i = 0

     while i < (len(ids)):
          if i+1 < len(ids):
               if ids[i] == pair[0] and ids[i+1] == pair[1]:
                    new_ids.append(swap_id)
                    i += 2
               else:
                    new_ids.append(ids[i])
                    i += 1
          else:
               new_ids.append(ids[i])
               i += 1
     return new_ids

class Tokenizer:

     def __init__(self):
          self.merges = {}
          self.vocabulary = {}

     def train(self, text, vocab_size):
          """
          realiza o treino para o tokenizador recebendo,
          como parâmetro, um texto e o tamanho do vocabulário
          """
          tokens = text.encode("utf-8") 
          tokens = list(map(int, tokens))
          ids = list(tokens)

          num_merges = vocab_size - 256
          merges = {}
          # Realiza os merges
          for i in range(num_merges):
               stats = get_stats(ids)
               if not stats:
                    break
               pair = max(stats, key=stats.get)
               idx = 256 + i
               ids = merge(pair, ids, idx)
               merges[pair] = idx

          # Cria e elabora o vocabulário
          vocabulary = {}
          for idx in range(256):
               vocabulary[idx] = bytes([idx])
          for (x, y), idx in merges.items():
               vocabulary[idx] = vocabulary[x] + vocabulary[y]

          self.vocabulary = vocabulary
          self.merges = merges

     def encode(self, text):
          """
          Dada uma string, retorna um lista de inteiros. Esta lista contém os tokens.
          Para isso é encontrado o par mínimo e realizando o merge. 
          """ 
          tokens = text.encode("utf-8")
          tokens = list(tokens)

          while len(tokens) >= 2:
               stats = get_stats(tokens)
               pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
               if pair not in self.merges:
                    break
               idx = self.merges[pair]
               tokens = merge(pair, tokens, idx)

          return tokens
     
     def decode(self, ids):
          """ Recebe uma lista de inteiro, para reverter para string """ 
          tokens = b"".join(map(self.vocabulary.__getitem__, ids))
          text = tokens.decode("utf-8", errors="replace")
          return text
     

     def showVocabulary(self):
          """ Mostra todas as subpalavras do vocabulário. """ 
          i = 256
          print("Subpalavras: ")
          for i in self.merges.values():
               subword = self.vocabulary.get(i)
               print(subword.decode("utf-8", errors="replace"))