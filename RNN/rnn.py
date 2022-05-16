import torch.nn as nn
import torch

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        #Remember that at the very beginning we concatenate the input and hidden vector
        #that's why size of the input is the sum of the sizes

        # hidden = tanh(Wx + Wh_-1) 
        self.input_to_hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        # output = sigmoid(Wh)
        self.hidden_to_input = nn.Linear(self.hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        #print(input.shape, hidden.shape)
        combined = torch.cat((input, hidden), dim=1) # dim = 0 is the batch size, dim = 1 are the actual values so I want dim = 1
        #print(combined.shape)

        hidden = self.tanh(self.input_to_hidden(combined)) # hidden_state at time t

        output = self.softmax(self.hidden_to_input(hidden)) #softmax to have probability, output

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)



if __name__ == '__main__':
    input = torch.randn((10, 32)) # one elemtn in a sequence
    hidden = torch.zeros((10, 64))

    rnn = RNN(32, 64, 50)

    out, hid = rnn(input,hidden)
    print(out.shape)
    