# author: Avik Ray (avik.r@samsung.com) 
#
# script modified from tensorflow ENG->FR machine translation using 
# sequence-to-sequence tutorial code by The TensorFlow Authors
#
# ==============================================================================
"""data utils"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import csv
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk import pos_tag
# nltk.download('all')

from tensorflow.python.platform import gfile
import tensorflow as tf

# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3


def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.append(space_separated_fragment) 
    
  return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
  """Create vocabulary file (if it does not exist yet) from data file.

  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  print("creating vocab from",data_path)
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="r") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 10000 == 0:
          print("  processing line %d" % counter)
        tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
        for w in tokens:
          word = w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
            
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]

      with gfile.GFile(vocabulary_path, mode="w") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + "\n")


def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="r") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
  """Convert a string to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

  Args:
    sentence: the sentence in bytes format to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

  Returns:
    a list of integers, the token-ids for the sentence.
  """

  if tokenizer:
    words = tokenizer(sentence)
  else:
    words = basic_tokenizer(sentence)
 
  return [vocabulary.get(w, UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
  """Tokenize data file and turn into token-ids using given vocabulary file.

  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  print("tokenizing file",data_path)
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    #with gfile.GFile(data_path, mode="rb") as data_file:
    with gfile.GFile(data_path, mode="r") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 1000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(line, vocab,tokenizer, normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_parse_data(data_dir, from_vocabulary_size, to_vocabulary_size, tokenizer=None):
  """create vocabularies and tokenize data.

  Args:
    data_dir: directory in which the data sets will be stored.
    from_vocabulary_size: size of the query vocabulary to create and use.
    to_vocabulary_size: size of the logical form vocabulary to create and use.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.

  Returns:
    A tuple of 6 elements:
      (1) path to the token-ids for query training data-set,
      (2) path to the token-ids for logical-form training data-set,
      (3) path to the token-ids for query development data-set,
      (4) path to the token-ids for logical-form development data-set,
      (5) path to the query vocabulary file,
      (6) path to the logical-form vocabulary file.
  """
  # Get wmt data to the specified directory.
  train_path = os.path.join(data_dir,"embed_train.txt")
  if FLAGS.valid:
    dev_path = os.path.join(data_dir,"embed_valid.txt")
  else:
    dev_path = os.path.join(data_dir,"embed_test.txt")

  from_train_path = train_path[:-4] + "_q.txt"
  to_train_path = train_path[:-4] + "_f.txt"
  from_dev_path = dev_path[:-4] + "_q.txt"
  to_dev_path = dev_path[:-4] + "_f.txt"
  return prepare_data(data_dir, from_train_path, to_train_path, from_dev_path, to_dev_path, from_vocabulary_size,
                      to_vocabulary_size, tokenizer)


def splitToFrom(data_dir,inputFile,out_key, id_arg=False):
    count = 0
    fromFile = out_key+"_q.txt"
    toFile = out_key+"_f.txt"
    fw_from = open(os.path.join(data_dir,fromFile),"w")
    fw_to = open(os.path.join(data_dir,toFile),"w")
    
    if inputFile.endswith(".csv"):
      with open(os.path.join(data_dir,inputFile)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        count=0
        for row in csv_reader:
          if count == 0:
            count+=1
            continue

          if id_arg:
            row[0], row[1], _ = replace_constants(row[0], row[1])
          
          fw_from.write(row[0]+"\n")
          fw_to.write(row[1]+"\n")
          count+=1
          
    else:
      fr = open(os.path.join(data_dir,inputFile),"r")
      count = 0
      for line in fr:
          line = line.strip().split("\t")
          if len(line)<2:
              continue
              
          query = line[0]
          lf = line[1]
          fw_from.write(query+"\n")
          fw_to.write(lf+"\n")
          count += 1
      fr.close()
          
    fw_from.close()
    fw_to.close()
      

    print("to-from split complete. number of lines =",count)    
    return

def replace_constants(f_from_line, f_to_line=None):
  if f_to_line:
    # find words in to
    target_words = re.findall(r'"(.*?)"', f_to_line)
    # replace constants with ids
    for ind,word in enumerate(target_words):
        f_to_line = f_to_line.replace('"'+word+'"', "arg"+str(ind))
        f_from_line = f_from_line.replace(word, "arg"+str(ind))
        print(f_to_line, f_from_line)

  else:
    tokens = nltk.word_tokenize(f_from_line)
    pos = nltk.pos_tag(tokens)

    # get the nouns
    target_words = [i[0] for i in pos if i[1] == 'NN']

  return f_from_line, f_to_line, target_words
    
def identify_constants (from_train_path, to_train_path=None):
    '''
    Identifies the function arguments that are meant to stay the same
    from the natural language command to the executable
    '''
    # open from file
    f_from = open(from_train_path,"r+")
    f_from_lines = f_from.readlines()

    #open to file
    if to_train_path:
      # open to file
      f_to = open(to_train_path,"r+")
      # f_to_lines = f_to.readlines()

      for i in range(len(f_to)):
        # find words in to
        target_words = re.findall(r'"(.*?)"', f_to_lines[i])
        # replace constants with ids
        for ind,word in enumerate(target_words):
            f_to_line = f_to_lines[i].replace('"'+word+'"', "arg"+str(ind))
            f_from_line = f_from_lines[i].replace(word, "arg"+str(ind))
            # print(f_to_line, f_from_line)
            f_from.seek(i)
            f_from.write(f_from_line)

            f_to.seek(i)
            f_to.write(f_to_line)
            f_from.truncate()
            f_from.close()

    else:
      # open arg file
      f_args = open("args.txt", "r+")

    for i in range(len(f_from_lines)):
      f_from_lines[i], f_to_lines[i], args = replace_constants(f_from_lines[i], f_to_lines[i])
      f_args.write(",".join(args)+"\n")

    # overwrite previous content
    if to_train_path:
      f_to.seek(0)
      f_to.writelines(f_to_lines)
      f_to.truncate()
      f_to.close()
    else:
      # close arg file
      f_args.close()

    f_from.seek(0)
    f_from.writelines(f_from_lines)
    f_from.truncate()
    f_from.close()

    # close arg file
    f_args.close()

def prepare_data(data_dir, from_train_path, to_train_path, from_dev_path, to_dev_path, from_vocabulary_size,
                 to_vocabulary_size, tokenizer=None):
  """Preapre all necessary files that are required for the training.

    Args:
      data_dir: directory in which the data sets will be stored.
      from_train_path: path to the file that includes "from" training samples.
      to_train_path: path to the file that includes "to" training samples.
      from_dev_path: path to the file that includes "from" dev samples.
      to_dev_path: path to the file that includes "to" dev samples.
      from_vocabulary_size: size of the "from language" vocabulary to create and use.
      to_vocabulary_size: size of the "to language" vocabulary to create and use.
      tokenizer: a function to use to tokenize each data sentence;
        if None, basic_tokenizer will be used.

    Returns:
      A tuple of 6 elements:
        (1) path to the token-ids for "from language" training data-set,
        (2) path to the token-ids for "to language" training data-set,
        (3) path to the token-ids for "from language" development data-set,
        (4) path to the token-ids for "to language" development data-set,
        (5) path to the "from language" vocabulary file,
        (6) path to the "to language" vocabulary file.
    """
  # Create vocabularies of the appropriate sizes.
  to_vocab_path = os.path.join(data_dir, "vocab%d.to" % to_vocabulary_size)
  from_vocab_path = os.path.join(data_dir, "vocab%d.from" % from_vocabulary_size)
  # create to vocab
  create_vocabulary(to_vocab_path, to_train_path, to_vocabulary_size, tokenizer)
  # create from vocab
  create_vocabulary(from_vocab_path, from_train_path, from_vocabulary_size, tokenizer)

  # Create token ids for the training data.
  to_train_ids_path = to_train_path + (".ids%d" % to_vocabulary_size)
  from_train_ids_path = from_train_path + (".ids%d" % from_vocabulary_size)
  data_to_token_ids(to_train_path, to_train_ids_path, to_vocab_path, tokenizer) # to
  data_to_token_ids(from_train_path, from_train_ids_path, from_vocab_path, tokenizer) # from

  # Create token ids for the development data.
  to_dev_ids_path = to_dev_path + (".ids%d" % to_vocabulary_size)
  from_dev_ids_path = from_dev_path + (".ids%d" % from_vocabulary_size)
  data_to_token_ids(to_dev_path, to_dev_ids_path, to_vocab_path, tokenizer)
  data_to_token_ids(from_dev_path, from_dev_ids_path, from_vocab_path, tokenizer)

  return (from_train_ids_path, to_train_ids_path,
          from_dev_ids_path, to_dev_ids_path,
          from_vocab_path, to_vocab_path)

          
def tokenize_dataset(from_data,to_data,from_vocab,to_vocab):

    dataset = []
    f_from = open(from_data,"r")
    for line in f_from:
        line = line.strip()
        if len(line)<1:
            continue
            
        token_ids = sentence_to_token_ids(line, from_vocab)
        dataset.append([token_ids])
       
    f_from.close()

    if to_data and to_vocab: 
      f_to = open(to_data,"r")
      idx = -1
      for line in f_to:
          line = line.strip()
          if len(line)<1:
              continue
              
          idx += 1
          token_ids = sentence_to_token_ids(line, to_vocab)
          dataset[idx].append(token_ids)
          
      f_to.close()
    
    return dataset
    