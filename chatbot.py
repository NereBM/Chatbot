#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 09:31:11 2020

@author: nereabejar
"""
import numpy as np
import tensorflow as tf
import re #to clean the text
import time

#########################################################################
##################     Creating a chatbot      ##########################
#########################################################################

#download the dataset from: https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
#This dataset contains thousands of conversations from movies
#We have to separate the metadata(the data that decribes the data we have), as we won't use it for our model
#We will use the movie_conversations.txt and the movie_lines.txt

###################    Auxiliar functions   ###################
def clean_text(text):
    text = text.lower() #all to lower caps
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text) # note the space 
    text = re.sub(r"\'ve", " have", text) # note the space 
    text = re.sub(r"\'re", " are", text) # note the space 
    text = re.sub(r"\'d", " would", text) # note the space 
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text) # all this characters I replacethem with nothing, because I want to delete them. Note the \ before the " so python knows I'm not ending the string
    return text


##################    PART 1 - Data pre-processing     ###########################

#import data:
lines = open('movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n') #we split by line breaks
conversations = open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n') #we split by line breaks
type(lines)
type(lines[0])
#..........      Dataset lines     ............
#'L1045 +++$+++ u0 +++$+++ m0 +++$+++ BIANCA +++$+++ They do not!'
#First column(L1045): Line id
#Second column(u0): user that say the line (this is not an id)
#third column: movie(m0): movie
#fourth column(BIANCA): user name
#fifth column: the line that they said
#..........      Dataset conversations     ............
#"u0 +++$+++ u2 +++$+++ m0 +++$+++ ['L194', 'L195', 'L196', 'L197']"
#First column(u0): user 0
#Second column(u2): user n
#Third column(m0): movie
#Fourth column: list of the lines id that belong to that conversation

# creating a dictionary that maps each line id with its content(the actual line)
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ') #the underscore at the beginning indicates that we create a temporary variable that will be destroy outside the loop
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]
    else:
        print("Error adding this element to dictionary")

#we want a list of the list of lines, so a list of conversations
conversations_lines = []
for conv in conversations:
    _conv = conv.split(" +++$+++ ")
    conversations_lines+=[_conv[-1][1:-1].replace("'","").replace(" ","").split(",")]

#The first line id of each of the conversations elements will be considered the question, and the second the answer
questions = []
answers = []
for conv in conversations_lines:
    for i in range(0,len(conv)-1):#we do it for every element but the last one, as the last one does not have an answer
        questions += [id2line[conv[i]]]
        answers += [id2line[conv[i+1]]]


#cleaning the questions and the answers:
clean_questions = []
for q in questions:
    clean_questions += [clean_text(q)]
 
# Cleaning the answers
clean_answers = []
for a in answers:
    clean_answers += [clean_text(a)]


#we need to count the number of ocurrencies with each word:
word2count = {}
for question in clean_questions:
    for word in question.split(): # this splits by default into words
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1


#We need to choose a threshold to filter out the less frequent words:
threshold = 20 #we choose a value that get rid of the 5% ish
questionswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold:
        questionswords2int[word] = word_number #assigns to the word a unique identifier
        word_number += 1
threshold_answers = 20
answerswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold_answers:
        answerswords2int[word] = word_number#assign to the word a unique identifier
        word_number += 1


# the next step is to create tokens that we are going to add to our dictionaries:
#Tokens explanation:
# * PAD --> your GPU (or CPU at worst) processes your training data in batches and all the sequences
#           in your batch should have the same length. If the max length of your sequence is 8, your sentence
#           My name is guotong1988 will be padded from either side to fit this length: My name is guotong1988
#           _pad_ _pad_ _pad_ _pad_
# * EOS --> end of sentence
# * UNK --> "unknown token" - is used to replace the rare words that did not fit in your vocabulary. So
#           we will replace with this token all the words that don't pass as frequency thershold
# * SOS  --> "start of sentence", so the first token which is fed to the decoder along with the though 
#           vector in order to start generating tokens of the answer

    
tokens =['<PAD>','<EOS>','<UNK>','<SOS>']
for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1
for token in tokens:
    answerswords2int[token] = len(answerswords2int) + 1
#for token in tokens:
#    questionswords2int[token] = len(questionswords2int)+1 #we assing ids to the tokens
#    answerswords2int[token] = len(answerswords2int)+1
    
#creating the inverse dictionary of answerswords2int(because we will need to do this operation a lot):
answersints2word = {w_i: w for w, w_i in answerswords2int.items()} #w_i is word integers, and w the word


# Adding the EOS token to the end of every answer in the clean_answers list
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>' #note the space

# Now we want to replace all the words in the questions and the answers for the equivalent integers
questions_into_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int['<UNK>'])
        else:
            ints.append(questionswords2int[word])
    questions_into_int.append(ints)
answers_into_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<UNK>'])
        else:
            ints.append(answerswords2int[word])
    answers_into_int.append(ints)
    


# Sorting the questions and answers by the length OF THE QUESTIONS. This will speed up and optimise our training because it will reduce the amount of padding
# within the training.
sorted_clean_questions = []
sorted_clean_answers = []
MAX_LENGTH = 25 #we are going to leave out the questions that are too long
for length in range(1, 25 + 1): 
    for i in enumerate(questions_into_int):
        if len(i[1]) == length:
            sorted_clean_questions+=[i[1]]
            sorted_clean_answers+= [answers_into_int[i[0]]]



############## PART 2 - Building a SEQ2SEQ model ################
            
#In tensorflow, all the variables are used in tensors, tensors are advanced arrays, more advanced than numpy arrays.
#This array is of a single type and allows faster computations in de Deep NN. Then, the first step is to go from
# the numpy arrays to tensors. After that, we need to define the variables used in tensors in tensorflow placeholders.

####################    Auxiliar functions   ###################
#.............   Creating the placeholders   ...............
def model_inputs():
    #tf.placeholder(type_of_data, dimension_of_the_matrix_of_the_input_data, name that we want to give to the inputs)
    inputs = tf.placeholder(tf.int32, [None, None], name = 'input')
    #dimension: our "sorted_clean_questions" is a 2-dimensional matrix [size, list of words]. we indicate this using [None, None]
    #dimension: our "sorted_clean_answers" is a 2-dimensional matrix [size, list of words]
    targets = tf.placeholder(tf.int32, [None, None], name = 'target')
    #now we need to create two more simple placeholders that will contain
    #  -  The learning rate(hyperparameter)
    #  -  The 'keep prob' - parameter that will control the dropout rate, which is the rate of the new runs you choose to overwrite during one iteration in the training. usually is 20%
        #Dropout is a regularization method that approximates training a large number of neural networks with different architectures in parallel.
        #During training, some number of layer outputs are randomly ignored or “dropped out.
    lr = tf.placeholder(tf.float32, name = 'learning_rate')  # no dimensions because it's a single element
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    return inputs, targets, lr, keep_prob


#.............   Pre-processing the targets   ...............
#The targets are the answers, and the RNN won't accept them one by one, but into batches.
#The targests must be feeded to the NN into batches. This means we will not pass the answers one by one, but into batches(eg. 10 by 10)
#each of the answers must start with the SOS token, and since we want to keep the size, we remove the last word(that was the EOS and we won't need it for the decoder)
def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], word2int['<SOS>']) # this creates a vector. The first argument is the dimensions[rows,column]. The second argument is the values we want to assign
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1]) #this function extracts a slice from a tensor
    # strided_slice() arguments:
    # First(target)
    # Second([0,0]) - from where we want to extract the slice [row,col]
    # Third([batch_size, -1]) - to where we want to extract the slice[row,col]. So we want the number of rows of the batch size, and all columns but the last one.
    # Fourth([1,1]) - indicates we want to slice one by one
    prepro_targets = tf.concat([left_side, right_side], 1)
    return prepro_targets

    
#.............   Creating the Encoder RNN layer   ...............
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    """
    This function created the encoder layer of the RNN
    
    Parameters:
        
        - rnn_inputs: inputs, targets and learning rate
        - rnn_size: number of input tensors of the encoder layer
        - num_layers: number of layers of the encoder
        - keep_prob: to control the dropout rate
        - sequence_length: the list of the lenght of each question in the batch
    
    """
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    _, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                                    cell_bw = encoder_cell,
                                                                    sequence_length = sequence_length,
                                                                    inputs = rnn_inputs,
                                                                    dtype = tf.float32)
    # tf.nn.bidirectional_dynamic_rnn returns two elements, and we only want the second one which is the encoder state. that's why we use: _, encoder_state
    # note that we don't use a simple rnn, we want it to be as powerfull as possible that's why we use bidirectional_dynamic_rnn, this is a dynamic version of a bidirectional RNN.
    # This version takes your input and will beuild independant forth and backwards RNNs.
    
    return encoder_state


#.............   Decoding the training set   ...............
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    """
    Parameters:
        - encoder_state: result from the tf.nn.bidirectional_dynamic_rnn()
        - decoder_cell: cell of the decoder
        - decoder_embedded_inputs: inputs of the decoder on whihch we apply embedding
        - sequence_length: the list of the lenght of each question in the batch
        - output_function:      the function we will ise to return the resutls of the decoder
        """
    # Embeding is a mapping from discrete objects, such as words, to vectors of real numbers.
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    # - attention_keys: keys to be compared to the target state
    # - attention_values: values that will be used to contruct the context vectors. The contexts is return by the encoder and that should be used by the decoder as the first element os the decoding
    # - attention_score_function: use to compute the similarity between the keys and the attention state
    # - attention_construct_function: used to build the attention state
    #the next step is to obtain the training decoderfunction, that will do the decoding of the training set
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = "attn_dec_train")
    decoder_output, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                   training_decoder_function,
                                                                   decoder_embedded_input,
                                                                   sequence_length,
                                                                   scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)


#.............   Decoding the test/validation set   ...............
    
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, decoding_scope, output_function, keep_prob, batch_size):
    """
        - encoder_state: result from the tf.nn.bidirectional_dynamic_rnn()
        - decoder_cell: cell of the decoder
        - decoder_embedding_matrix: 
        - sos_id: SOS token id
        - eos_ir: EOS token id
        - max_length: max size of the inputs
        - num_words: number of words od the dictionary that we have that contains all the words 
        - sequence_length: the list of the lenght of each question in the batch
        - output_function: the function we will ise to return the resutls of the decoder
    """
    # Embeding is a mapping from discrete objects, such as words, to vectors of real numbers.
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size]) # initialises the attention states to zeros
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name = "attn_dec_inf")
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                test_decoder_function,
                                                                                                                scope = decoding_scope)
    return test_predictions
    
#.............   Creating the Decoder RNN  ...............
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev = 0.1) #we initialised the weights. this functions originates a truncated normal distribution
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weights,
                                                                      biases_initializer = biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables() #take our dedcoding scope and
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
    return training_predictions, test_predictions
 

#Building the seq2seq model
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    """
    This function assembles (put together) the encoder that returns the encoder states and the decoder
    that returns the the training and test predictions
    """
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1, #we add a plus one because the upper bound in the sequence is excluded
                                                              encoder_embedding_size, #number of dimensions of the encoder embedding matrix
                                                              initializer = tf.random_uniform_initializer(0, 1)) # this will give us numbers from 0 to 1
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionswords2int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions


##########      PART 3 - TRAINING THE MODEL    ##########
    
# ..............   Hyperparameters   ................
epochs =100
batch_size = 64
rnn_size = 512
num_layers = 3
encoding_embedding_size = 512  #is the number of columns in your embedding matrix, that is the number of columns that you want to have for the embedding values
decoder_embedding_size = 512
learning_rate = 0.01
learning_rate_decay = 0.9 # one of the most common values is 90%
min_learning_rate = 0.0001
keep_probability = 0.5  # value recommended in the following paper: "Dropping out 20% of the inputs units and 50% of the hidden units is often found to be optimal"
#check this paper about how the dropout prevents NN from overfitting: 
#https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf

#...............   Defining a session   ...............
#The first thing to do when defining a session is to reset the default tensor flow graph
tf.reset_default_graph()
session = tf.InteractiveSession()

#.............   Loading the model inputs   .............
inputs, targets, lr, keep_prob = model_inputs()

#.............   Setting the sequence length   ..........
#we will set it to 25 which is the maximum lenth:
sequence_length = tf.placeholder_with_default(25, None, name='sequence_length') #with default because we're going to set a default function to use when this parameter is not fed to the NN
#the 25 value means that in the training we won't be using the questions and answers that have more than 25 words.

#..............  Getting the shape of the inputs tensor  .........
inputs_shape = tf.shape(inputs)

#..............  Getting the training and test predictions  .........
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                       targets,
                                                       keep_probability,
                                                       batch_size,
                                                       sequence_length,
                                                       len(answerswords2int),
                                                       len(questionswords2int),
                                                       encoding_embedding_size,
                                                       decoder_embedding_size,
                                                       rnn_size,
                                                       num_layers,
                                                       questionswords2int
                                                       )


#........... Loss Error, Optimiser and Gradient Clipping
#Loss error: we'll use the weighted cross entropy loss error (best suited when using sequences and deep NLP in general)
#Optimiser: first an Adam optimiser (one of the best for gradient descent), 
#Gradient Clipping: is a technique that will cut the gradient in the graph between the minimum value 
#and the maximum value to avoid exploding or vanishing gradient issues
with tf.name_scope("optimisation"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                  targets,
                                                  tf.ones([inputs_shape[0],sequence_length])) #this inicialises the weights with ones
    optimiser = tf.train.AdamOptimizer(learning_rate)
    gradients = optimiser.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None]
    optimiser_gradient_clipping = optimiser.apply_gradients(clipped_gradients)
    
#............   Applying the padding to the questions and answers   .......
# Why? Because all the sentences(Qs ans As) must have the same length
# How? The <PAD> elements are added so that the question sequence and the answer sequence have the same length
# This is one example:
#  - Question: ['Who', 'are', 'you']
#  - Answer: [ <SOS>, 'I', 'am', 'a', 'bot', '.', <EOS>]
#They don't have the same length, but after applying the padding they will turn into:
#  - Question: ['Who', 'are', 'you', <PAD>, <PAD>, <PAD>, <PAD>]
#  - Answer: [ <SOS>, 'I', 'am', 'a', 'bot' '.', <EOS>, <PAD>]
#So we will do this for each sentence in each batch:
def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(seq) for seq in batch_of_sequences])
    return [ seq + [word2int['<PAD>']]*(max_sequence_length - len(seq)) for seq in batch_of_sequences]

#............     Splitting data into batches of questions and answers    ........
def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, len(questions) // batch_size):
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index : start_index + batch_size]
        answers_in_batch = answers[start_index : start_index + batch_size]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questionswords2int))#we need numpy arrays because we're working with tensorflow
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, answerswords2int))
        yield padded_questions_in_batch, padded_answers_in_batch #yield is as a return. We use yield when we want to iterate over a sequence, but don’t want to store the entire sequence in memory.

#............    Splitting Qs and As into training and testing sets    .............
training_validation_split = int(len(sorted_clean_questions)* 0.15)
training_questions = sorted_clean_questions[training_validation_split:]
training_answers = sorted_clean_questions[training_validation_split:]
validation_questions = sorted_clean_questions[:training_validation_split]
validation_answers = sorted_clean_questions[:training_validation_split]


#...............   Training    ..........
batch_index_check_training_loss = 100
batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) - 1
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 1000
checkpoint = "chatbot_weights.ckpt" # For Windows users, replace this line of code by: checkpoint = "./chatbot_weights.ckpt"
session.run(tf.global_variables_initializer())
for epoch in range(1, epochs + 1):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimiser_gradient_clipping, loss_error], {inputs: padded_questions_in_batch,
                                                                                               targets: padded_answers_in_batch,
                                                                                               lr: learning_rate,
                                                                                               sequence_length: padded_answers_in_batch.shape[1],
                                                                                               keep_prob: keep_probability})
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches: {:d} seconds'.format(epoch,
                                                                                                                                       epochs,
                                                                                                                                       batch_index,
                                                                                                                                       len(training_questions) // batch_size,
                                                                                                                                       total_training_loss_error / batch_index_check_training_loss,
                                                                                                                                       int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
                batch_validation_loss_error = session.run(loss_error, {inputs: padded_questions_in_batch,
                                                                       targets: padded_answers_in_batch,
                                                                       lr: learning_rate,
                                                                       sequence_length: padded_answers_in_batch.shape[1],
                                                                       keep_prob: 1})
                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_questions) / batch_size)
            print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error, int(batch_time)))
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print('I speak better now!!')
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print("Sorry I do not speak better, I need to practice more.")
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    if early_stopping_check == early_stopping_stop:
        print("My apologies, I cannot speak better anymore. This is the best I can do.")
        break
print("Game Over")
















