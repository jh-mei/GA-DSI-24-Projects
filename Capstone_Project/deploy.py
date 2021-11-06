import streamlit as st
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import re
import numpy as np
from torch.utils.data import Dataset
import csv
from transformers import (GPT2Tokenizer,
                          GPT2LMHeadModel)

#===========================================#
#              Load LSTM Model              #
#===========================================#

# model definition
class CharRNN(nn.Module):
    def __init__(self, tokens, n_hidden=256, n_layers=2, drop_prob=0.5):
        super().__init__()  # access inheritance for superclass
        self.drop_prob = drop_prob
        self.n_layers = n_layers  # recurrent layer number (number of layers)
        self.n_hidden = n_hidden  # hidden feature number (nodes)

        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        self.dropout = nn.Dropout(drop_prob)  # instantiate dropout layer
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers,
                            dropout=drop_prob, batch_first=True)  # instantiate lstm hidden layers
        self.fc = nn.Linear(n_hidden, len(self.chars))  # output layer with len(self.chars) as number of output nodes

    # forward propagation
    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)
        x = self.dropout(x)
        x = self.fc(x)

        return x, hidden

    # make a class prediction
    def predict(self, char, hidden=None, device=torch.device('cpu'), top_k=None):
        with torch.no_grad():
            self.to(device)
            try:
                x = np.array([[self.char2int[char]]])  # create array of matrices
            except KeyError:
                return '', hidden

            x = one_hot_encode(x, len(self.chars))  # one hot encode array of matrices
            inputs = torch.from_numpy(x).to(device)

            out, hidden = self.forward(inputs, hidden)

            p = F.softmax(out, dim=2).data.to('cpu')

            if top_k is None:
                top_ch = np.arange(len(self.chars))
            else:
                p, top_ch = p.topk(top_k)
                top_ch = top_ch.numpy().squeeze()

            if top_k == 1:
                char = int(top_ch)
            else:
                p = p.numpy().squeeze()
                char = np.random.choice(top_ch, p=p / p.sum())

            return self.int2char[char], hidden

def one_hot_encode(arr, n_labels):
    # create matrices for each value in array with n_labels number of 0s
    one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)

    # populate the matrix with ones at indices specified by values of array
    one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.

    # reshape it to get back to the size of original array
    one_hot = one_hot.reshape((*arr.shape, n_labels))

    return one_hot

def load_checkpoint(filename):
    with open(filename, 'rb') as f:
        checkpoint = torch.load(f, map_location='cpu')

    net = CharRNN(checkpoint['tokens'], n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
    net.load_state_dict(checkpoint['state_dict'])

    return net, checkpoint

def sample_lines(net, n_lines=3, prime='import', top_k=None, device='cpu', max_len=100):
    net.to(device)
    net.eval()

    # First off, run through the prime characters
    chars = []
    h = None
    for ch in prime:
        char, h = net.predict(ch, h, device=device, top_k=top_k)

    chars.append(char)

    # Now pass in the previous character and get a new one
    l = 0
    for ii in range(max_len):
        char, h = net.predict(chars[-1], h, device=device, top_k=top_k)
        chars.append(char)
        if char == '\n':
            l += 1
            if l == n_lines:
                break

    return ''.join(chars)

def clean_string(x):
    x = re.sub('#.*$', '', x, flags=re.MULTILINE)
    x = re.sub("'''[\s\S]*?'''", '', x, flags=re.MULTILINE)
    x = re.sub('"""[\s\S]*?"""', '', x, flags=re.MULTILINE)
    x = re.sub('^[\t]+\n', '', x, flags=re.MULTILINE)
    x = re.sub('^[ ]+\n', '', x, flags=re.MULTILINE)
    x = re.sub('\n[\n]+', '\n\n', x, flags=re.MULTILINE)
    return x

def sample(line, prime):
    clean_prime = clean_string(prime)
    print(sample_lines(net, 1, prime=clean_prime, top_k=3))


#===========================================#
#              Markov Chains                #
#===========================================#


# create stochastic matrix function
def create_matrix(corpus, k=4):  # k is chain order, default 4
    T = {}  # empty matrix

    for i in range(len(corpus) - k):
        X = corpus[i:i + k]  # slice k characters
        Y = corpus[i + k]  # the character after X

        if T.get(X) is None:  # if X does not exist in matrix yet
            T[X] = {}  # create X key
            T[X][Y] = 1  # create 1 instance of Y after X
        else:
            if T[X].get(Y) is None:  # otherwise if Y value does not exist for X key
                T[X][Y] = 1  # create 1 instance of Y after X
            else:  # otherwise...
                T[X][Y] += 1  # add 1 instance of Y after X

    return T


# convert frequency from stoc matrix to probabilities
def freq2prob(T):
    for kx in T.keys():
        s = float(sum(T[kx].values()))  # sum of total frequencies
        for k in T[kx].keys():
            T[kx][k] = T[kx][k] / s  # probability of frequency

    return T

# run model
def model(corpus, k=4):
    T = create_matrix(corpus, k)
    T = freq2prob(T)
    return T


def sample_next(char, model, k):
    char = char[-k:]
    if markov_model.get(char) is None:  # if char not found in matrix
        return " "

    possible_chars = list(markov_model[char].keys())  # retrieve key from stoch matrix
    possible_values = list(markov_model[char].values())  # retrieve value from stoch matrix

    return np.random.choice(possible_chars, p=possible_values)

def generate_text(sentence, k=4, max_len=5):
    char = sentence[-k:]
    for ix in range(max_len):
        next_prediction = sample_next(char, markov_model, k)
        sentence += next_prediction
        char = sentence[-k:]
    return sentence

#===========================================#
#                  GPT-2                    #
#===========================================#

# instantiate dataset class
class email(Dataset):

    def __init__(self, truncate=False, gpt2_type='gpt2', max_length=768):

        # instantiate pretrained tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.emails = []

        with open('https://github.com/jh-mei/GA-DSI-24-Projects/blob/master/Capstone_Project/dataset/enron6_clean.csv', newline='') as csvfile:
            email_csv = csv.reader(csvfile)
            for row in email_csv:
                # encode text into tensors
                self.emails.append(torch.tensor(
                    self.tokenizer.encode(
                        # 768 characters is gpt2-small's limit
                        # endoftext is gpt2 specific delimiter
                        f'{row[0][:max_length]}<|endoftext|>'
                    )))

        if truncate:
            self.emails = self.emails[:20000]

        self.email_count = len(self.emails)

    def __len__(self):
        return self.email_count

    def __getitem__(self, item):
        return self.emails[item]  # return a particular tensor


# ensure each input tensors have as much text as possible
def pack_tensor(new_tensor, packed_tensor, max_seq_len):
    if packed_tensor is None:
        return new_tensor, True, None
    if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
        return packed_tensor, False, new_tensor
    else:
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
        return packed_tensor, True, None


# adapted from Huggingface's run_generation.py script
def generate(
        model,
        tokenizer,
        prompt,
        entry_count=1,
        entry_length=100,
        top_p=0.8,
        temperature=1.,
):
    model.eval()

    generated_num = 0

    filter_value = -float('Inf')

    with torch.no_grad():

        for entry_idx in range(entry_count):

            entry_finished = False

            generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

            for i in range(entry_length):
                outputs = model(generated, labels=generated)
                loss, logits = outputs[:2]
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)

                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value

                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)

                if next_token in tokenizer.encode('<|endoftext|>'):
                    entry_finished = True

                if entry_finished:
                    generated_num = generated_num + 1

                    output_list = list(generated.squeeze().numpy())
                    output_text = tokenizer.decode(output_list)

                    break

            if not entry_finished:
                output_list = list(generated.squeeze().numpy())
                output_text = f'{tokenizer.decode(output_list)}<|endoftext|>'

    return output_text




#===========================================#
#              Streamlit Code               #
#===========================================#


st.header('Capstone - Financial Email Autocomplete')
st.caption('by Mei Junhao')

df = pd.DataFrame({
  'first column': ['LSTM', 'Markov Chain', 'GPT-2']
})

option = st.selectbox(
    'Select Model',
     df['first column'])

if option=='LSTM':
    'LSTM is decent at text generation. Just don\'t expect them to make too much sense.'
elif option=='Markov Chain':
    'Markov Chains model is alright at autocompleting incomplete words, eg. "Regar" -> "Regards". It might not give you the desired output on the first try, so in that case, try again!'
else:
    'GPT-2 might give you something unrelated unless you are specific about the topic in the input text. This might take a while to run.'

if option=='LSTM' or option=='Markov Chain':
    char_len = st.number_input('No. of chars to generate', min_value=1, max_value=None)

input = st.text_area('Seed text (required)', value="Enter text here", key='input')

if option=='LSTM':
    if st.button('Generate!', key='lstm'):
        net, _ = load_checkpoint('https://github.com/jh-mei/GA-DSI-24-Projects/blob/master/Capstone_Project/models/lstm_model.pt')
        prime = 'test'
        clean_prime = clean_string(prime)
        clean_prime = clean_string(input)
        generated_text = sample_lines(net, 1, prime=clean_prime, top_k=3, max_len=char_len)
        st.write(generated_text)

elif option=='Markov Chain':
    if st.button('Generate!', key='mm'):
        ran_mmodel = False
        if ran_mmodel == False:
            with open('https://github.com/jh-mei/GA-DSI-24-Projects/blob/master/Capstone_Project/dataset/enron6_clean.txt', 'r') as f:
                corpus = f.read()
            corpus = corpus.replace('\n', ' ').replace('?', '.').replace('!', '.').replace('“', '.').replace('”', '.').replace('/', ' ').replace('‘', ' ').replace('-', ' ').replace('’', ' ').replace('\'', ' ').replace('=', ' ').replace('\\', ' ').replace('_', ' ')
            markov_model = model(corpus)
            generated_text = generate_text(input, k=4, max_len=char_len)
            st.write(generated_text)
        else:
            generated_text = generate_text(input, k=4, max_len=char_len)
            st.write(generated_text)


else:
    if st.button('Generate!', key='gpt'):
        ran_model = False
        if ran_model == False:
            gptmodel = GPT2LMHeadModel.from_pretrained('gpt2')
            gptmodel.load_state_dict(torch.load('https://github.com/jh-mei/GA-DSI-24-Projects/blob/master/Capstone_Project/models/gpt2_10epochs.pt', map_location='cpu'))
            gptmodel.eval()
            ran_model = True
            generated_text = generate(gptmodel.to('cpu'), GPT2Tokenizer.from_pretrained('gpt2'), input, entry_count=1)
            st.write(generated_text)
        else:
            generated_text = generate(gptmodel.to('cpu'), GPT2Tokenizer.from_pretrained('gpt2'), input, entry_count=1)
            st.write(generated_text)





