import torch
import torch.nn as nn
import torch.nn.functional as FF

class QuasiLayer(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size):
        super(QuasiLayer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size

        #kernel_size specifies on how many elements we want to look back
        # we need hidden_size for the output
        #we need 2 * hidden_size for the gates
        self.conv = nn.Conv1d(in_channels = input_size, out_channels = 3*hidden_size, kernel_size = kernel_size)


    def conv_step(self, sequence):
        #inputs have sape batch_size x length x hidden_size and we want to convolve about the length of the sequence so it must be at the end

        sequence = sequence.transpose(1, 2) # so then we have length at the end
        sequence = FF.pad(sequence, (self.kernel_size-1, 0)) # pad by how many and at which dimension
        gates = self.conv(sequence).transpose(1,2) #come back to previous shape

        #This is superior as we can get all the gates values as in the LSTM but
        #we can do it in paraller for all the data points
        #Z is a candidate vector, F is a forget gate, and o is a output gate
        Z, F, O = gates.split(split_size = self.hidden_size, dim=2)

        return Z.tanh(), F.sigmoid(), O.sigmoid()

    def rnn_step(self, z, f, o, c=None):
      

        #At the first iteration it is likely for c to be one. Then we use entirely candidate values
        #otherwise we mix old state with new candidate in the proportions proposed by the forget gate
        c = (1 - f) * z if c is None else f * c + (1 - f) * z
        h = o * c
        return c, h # we return current cell state and the actuall hiden we will return
    

    def forward(self, input_sequence):
        #inputs must have batch_size x length x attributes

        #But we still need some kind of recursive relation
        #c is some kind of context vector as in the lstm
        c_t = None # we start from empty context

        Z, F, O = self.conv_step(input_sequence)
        #print(Z.shape)
        hidden_states = []

        for z,f,o in zip(Z.split(1, dim=1), F.split(1, dim=1), O.split(1, dim=1)):
            c_t = (1 - f) * z if c_t is None else f * c_t + (1 - f) * z #elementwise multiplication
            h_t = o * c_t

            hidden_states.append(h_t)
        # So I have t hidden states and each is batch_size x actual_values
        #print(hidden_states[0].shape)
        hidden_states = torch.cat(hidden_states, dim=1).contiguous()

        return hidden_states, h_t 


        

if __name__ == "__main__":
    quasiRNN = QuasiLayer(20, 40, 3) # kernel_size 3 means that we will look at the 3 past tokens to create current states

    sequence = torch.randn((16, 4, 20))

    #for s in sequence.split(1, dim=1):
    #    print(s.shape)

    hidden_states, hidden_last = quasiRNN(sequence)
    print(hidden_states.shape)