'''Sequence to sequence example in Keras

English to Tagalog sentence pairs.
http://www.manythings.org/anki/tgl-eng.zip

Lots of neat sentence pairs datasets can be found at:
http://www.manythings.org/anki/

# References

- Sequence to Sequence Learning with Neural Networks
    https://arxiv.org/abs/1409.3215
- Learning Phrase Representations using
    RNN Encoder-Decoder for Statistical Machine Translation
    https://arxiv.org/abs/1406.1078
'''
from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
import numpy as np
from pyfasttext import FastText
from keras.utils import plot_model

from os import listdir
import random

def read_data(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = [content[i].split() for i in range(len(content))]
    content = np.array(content)
    content = np.reshape(content, [-1, ])
    return content

def build_dicts(words):
    dictionary = dict()
    for word in words:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

def get_words(sentences):
    words = []
    for sen in sentences:
        tokens = sen.split()
        for token in tokens:
            if token not in words:
                words.append(token)
    #print(len(words))
    return words

def max_wordnum(texts):
    count = 0
    for text in texts:
        if len(text.split()) > count:
            count = len(text.split())
    return count

def input2target(data_path, sos, eos):
    input_texts = []
    target_texts = []

    filename_list = listdir(data_path)

    for i in range(len(filename_list)):
        with open(data_path+'/'+filename_list[i], 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
            #print('length_lines'+str(len(lines)))
        for line in lines:
            if len(line) <= 0:
                #print('zero')
                continue
            line = line.replace(",", " ,")
            line = line.replace(".", " .")
            line = line.replace("!", " !")
            line = line.replace("?", " ?")
            line = line.lower()
            target_text, input_text = line.split('\t')
            # print(input_text , " : ", target_text)
            target_text = "%s %s %s" % (sos, target_text, eos)
            input_texts.append(input_text)
            target_texts.append(target_text)

    return input_texts, target_texts

def dataloader(data_path,enc_word2vec, dec_word2vec, batch_size, sos, eos,word_vec_size):

    #filename_list = listdir(data_path)
    #print('dataloader')
    input_texts, target_texts = input2target(data_path, sos, eos)
    '''
    print('in and target')
    print(input_texts[0])
    print(target_texts[0])
    print(len(input_texts[0]))
    print(len(target_texts[0]))
    '''
    #print('length_input_text',str( len(input_texts)))
    #print('length_target_text',str( len(target_texts)))
    #input_words = get_words(input_texts)
    #input_dict, input_rev_dict = build_dicts(input_words)

    #target_words = get_words(target_texts)
    #if sos in target_words:
    #    print("Present")
    #del(target_words)
    while True:
        #print('true')
        temp = list(zip(input_texts,target_texts))
        random.shuffle(temp)
        input_texts, target_texts = zip(*temp)
        del(temp)

        num_batches = int(len(input_texts)/batch_size)

        max_encoder_seq_length = max([len(words.split()) for words in input_texts])
        max_decoder_seq_length = max([len(words.split()) for words in target_texts])
        print('encoder decoder length')
        print(max_encoder_seq_length)
        print(max_decoder_seq_length)

        #print(image_list[(-2*batch_size)])
        ''' split input text '''
        input_texts_split = np.array_split(input_texts,num_batches)
        target_texts_split = np.array_split(target_texts, num_batches)

        while input_texts_split:
            input_texts_batch = input_texts_split.pop()
            target_texts_batch = target_texts_split.pop()



            #print('Number of samples:', len(input_texts))
            encoder_input_data = np.zeros(
                        (len(input_texts_batch), max_encoder_seq_length, word_vec_size),
                            dtype='float32')
            decoder_input_data = np.zeros(
                        (len(input_texts_batch), max_decoder_seq_length, word_vec_size),
                            dtype='float32')
            decoder_target_data = np.zeros(
                        (len(input_texts_batch), max_decoder_seq_length, word_vec_size),
                            dtype='float32')


            for i, text, in enumerate(input_texts_batch):
                words = text.split()
                for t, word in enumerate(words):
                    encoder_input_data[i, t, :] = enc_word2vec.get_numpy_vector(word, normalized=True)

            for i, text, in enumerate(target_texts_batch):
                words = text.split()
                for t, word in enumerate(words):
                    # decoder_target_data is ahead of decoder_input_data by one timestep
                    #decoder_input_data[i, t, target_dict[word]] = 1.
                    decoder_input_data[i, t, :] = dec_word2vec.get_numpy_vector(word, normalized=True)
                    if t > 0:
                        # decoder_target_data will be ahead by one timestep
                        # and will not incint(valid_sentences//batch_size)lude the start character.
                        #decoder_target_data[i, t - 1, target_dict[word]] = 1.
                        decoder_target_data[i, t - 1, :] = dec_word2vec.get_numpy_vector(word, normalized=True)

            #print('shapes')
            #print(encoder_input_data.shape)
            #print(decoder_input_data.shape)
            #print(decoder_target_data.shape)

            yield [encoder_input_data,decoder_input_data], decoder_target_data

def build_model(word_vec_size, latent_dim):
    # Path to the data txt file on disk.
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, word_vec_size))
    encoder = LSTM(latent_dim, return_sequences=True)(encoder_inputs)
    encoder_outputs, state_h, state_c = LSTM(latent_dim, return_state=True)(encoder)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, word_vec_size))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(word_vec_size, activation='tanh')
    #decoder_dense = Dense(word_vec_size)
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    encoder_model = Model(encoder_inputs, encoder_states)
    encoder_model.summary()

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model

def main():
    #enc_word2vec = FastText('wiki.tl/wiki.tl.bin')
    #dec_word2vec = FastText('wiki.en/wiki.en.bin')

    dec_word2vec = FastText('wiki.en/wiki.en.bin')
    enc_word2vec = FastText('wiki.tl/wiki.tl.bin')

    #data_path = 'tgl-eng/tgl.txt'
    data_path = 'train_split'
    valid_path = 'valid_split'
    #data_path = 'health_shortened.tsv'
    eos = "eos"
    sos = "sos"
    savemodel_filename = 's2s_fasttextloader_combineddata.h5'
    #training parameters
    #batch_size = 64  # Batch size for training.
    batch_size = 64
    epochs = 100  # Number of epochs to train for.
    latent_dim = 512 # Latent dimensionality of the encoding space.
    word_vec_size = 300
    num_sentence = 77989
    steps_per_epoch = int(num_sentence//batch_size)
    #steps_per_epoch = 2
    valid_sentences= 19498
    validation_steps = int(valid_sentences//batch_size)
    #validation_steps = 1
    chkpt_path="checkpoints/weights_cosine_proximity_combined2-{epoch:05d}.hdf5"
    checkpoint = ModelCheckpoint(chkpt_path, verbose=1)


    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model, encoder_model, decoder_model = build_model(word_vec_size, latent_dim)
    model.summary()

    # Compile & run training
    #model.compile(optimizer='rmsprop', loss='mean_squared_error')
    model.compile(optimizer=RMSprop(0.001), loss='cosine_proximity')
    #model.compile(optimizer='rmsprop', loss='mean_squared_error')
    # Note that `decoder_target_data` needs to be one-hot encoded,
    # rather than sequences of integers like `decoder_input_data`!


    model.fit_generator(dataloader(data_path,enc_word2vec, dec_word2vec ,batch_size, sos, eos,word_vec_size),steps_per_epoch=steps_per_epoch, validation_data =dataloader(valid_path,enc_word2vec, dec_word2vec ,batch_size, sos, eos,word_vec_size),
              epochs=epochs,validation_steps=validation_steps, verbose =1, callbacks=[checkpoint])

    plot_model(model, to_file='model_seq2seq.png',show_shapes=True)


    # Next: inference mode (sampling).
    # Here's the drill:
    # 1) encode input and retrieve initial decoder state
    # 2) run one step of decoder with this initial state
    # and a "start of sequence" token as target.
    # Output will be the next target token
    # 3) Repeat with the current target token and current states

    # Define sampling models
    print("Training finished.")
    '''
    decoder_model.summary()

    def decode_sequence(input_seq,sos,eos):
        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, word_vec_size))
        # Populate the first character of target sequence with the start character.
        #target_seq[0, 0, target_dict[sos]] = 1.


        target_seq[0,0,:] = dec_word2vec.get_numpy_vector(sos, normalized=True)

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict(
                [target_seq] + states_value)

            # Sample a token
            #sampled_token_index = np.argmax(output_tokens[0, -1, :])
            #sampled_word = target_rev_dict[sampled_token_index]
            sampled_word = dec_word2vec.words_for_vector(output_tokens[0,-1,:])[0][0]
            decoded_sentence += sampled_word + " "

            # Exit condition: either hit max length
            # or find stop character.
            # if sampled_word in [".", "?", "!"] or
            if (sampled_word == eos or
               len(decoded_sentence) > max_decoder_seq_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, word_vec_size))
            target_seq[0,0,:] = dec_word2vec.get_numpy_vector(sampled_word, normalized=True)

            # Update states
            states_value = [h, c]

        return decoded_sentence


    input_texts, target_texts = input2target(data_path, sos, eos)
    #target_texts_split = np.array_split(target_texts, steps_per_epoch)
    #input_texts_split = np.array_split(input_texts,steps_per_epoch)
    #input_texts_batch = input_texts_split.pop()
    #target_texts_batch = target_texts_split.pop()

    indexes = np.random.randint(0, len(input_texts), 50)
    max_encoder_seq_length = max([len(words.split()) for words in input_texts])
    max_decoder_seq_length = max([len(words.split()) for words in target_texts])
    encoder_input_data = np.zeros(
                (len(input_texts), max_encoder_seq_length, word_vec_size),
                    dtype='float32')
    for i, text, in enumerate(input_texts):
        words = text.split()
        for t, word in enumerate(words):
            encoder_input_data[i, t, :] = enc_word2vec.get_numpy_vector(word, normalized=True)


    for seq_index in indexes:
        # Take one sequence (part of the training set)
        # for trying out decoding.
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(input_seq,sos,eos)
        print('-')
        print('Input sentence:', input_texts[seq_index])
        print('Decoded sentence:', decoded_sentence)
    '''
if __name__ == '__main__':
    main()
