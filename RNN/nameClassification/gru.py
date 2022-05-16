import torch
import torch.nn as nn

class MyGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        Args:
            input_size (int): dimensionality of our input x_i has. This is not length of the sequence. 
                              This is by how many attributes given input is described
            output_size (int): dimensionality of our hidden and cell states have
        """
        super(MyGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.reset_gate_weights = nn.Linear(self.input_size + self.hidden_size + 2, self.hidden_size)

        self.update_gate_weights = nn.Linear(self.input_size + self.hidden_size + 2, self.hidden_size) # +1 comes from additional bias. In the implementations I saw that most usually 
    
        self.new_gate_weights_input = nn.Linear(self.input_size + 1, self.hidden_size)
        self.new_gate_weights_hidden = nn.Linear(self.hidden_size + 1, self.hidden_size)

    def forward(self, input_sequence, initial_hidden=None):
        """_summary_

        Args:
            input_sequence (tensor): tensor of shape Length_of_seq x batch_size x input_size
            initial_hidden (tuple of tensors, optional): Tuple of first hidden and cell states. Defaults to None.
        """
        hidden_states = [] # we will pack all hidden states for each timestep

        seq_length, batch_size, _ = input_sequence.shape

        if initial_hidden is None:
            hidden_t = torch.zeros(batch_size, self.hidden_size).to(input_sequence.device)              
        else:
            hidden_t = initial_hidden

        for t in range(seq_length):
            x_t = input_sequence[t, :, :] #take t-th element from every batch
            print(f"No concatenation. shape of input: {x_t.shape}, shape of hidden: {hidden_t.shape}, bias shape: 1")
            x_t_bias = torch.cat((x_t, torch.ones((batch_size, 1))), dim=1)
            hidden_t_bias = torch.cat((hidden_t, torch.ones((batch_size,1))), dim=1)
            input_t = torch.cat((x_t_bias, hidden_t_bias), dim=1)

            #print(input_t)
            print(input_t.shape)

            reset_gate_t = torch.sigmoid(self.reset_gate_weights(input_t)) # What I want to forget from previous hidden state
            update_gate_t = torch.sigmoid(self.update_gate_weights(input_t)) # What I want to update and keep from previous state
            new_gate_t = torch.tanh(self.new_gate_weights_input(x_t_bias) + reset_gate_t * self.new_gate_weights_hidden(hidden_t_bias)) # new information to be written
            hidden_t = (1 - update_gate_t) * new_gate_t + update_gate_t * hidden_t
            
            hidden_states.append(hidden_t.unsqueeze(0)) # we want this to have dimensions 1 x batch_size x dimensionality

        #at this time hidden_states is just a matrix of tensors it would be better to make it just a tensor
        hidden_states = torch.cat(hidden_states, dim=0).contiguous() # we concatenate along dimension when everything has 1 it will produce length_of_sequence as that many hidden_states we have

        return hidden_states, hidden_t


if __name__ == "__main__":
    input_dim = 10
    hidden_dim = 20
    sequence_length = 2
    batch_size = 2

    lstm = MyGRU(input_dim, hidden_dim)

    X = torch.randn((sequence_length, batch_size, input_dim))

    hidden_states, last_states = lstm(X)
    print(hidden_states.shape)

