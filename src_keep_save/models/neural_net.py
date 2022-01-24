from abc import ABC  # might not have any function

import torch
import torch.nn.functional as f
import torch.nn as nn


class PcNet(nn.Module):
    def __init__(self, input_size_prot=1024, input_size_comp=196, hidden_size_prot=32):
        super(PcNet, self).__init__()
        self.fc_prot = nn.Linear(input_size_prot, hidden_size_prot)
        self.fc_lin1 = nn.Linear(hidden_size_prot + input_size_comp, 1024)
        self.fc_drop1 = nn.Dropout(0.1)
        self.fc_lin2 = nn.Linear(1024, 1024)
        self.fc_drop2 = nn.Dropout(0.1)
        self.fc_lin3 = nn.Linear(1024, 512)
        self.fc_lin4 = nn.Linear(512, 1)

    def forward(self, x):
        out = f.relu(self.fc_prot(x[0]))
        out = torch.cat((out, x[1]), dim=1)
        out = self.fc_drop1(f.relu(self.fc_lin1(out)))
        out = self.fc_drop2(f.relu(self.fc_lin2(out)))
        out = f.relu(self.fc_lin3(out))
        out1 = f.relu(self.fc_lin4(out))
        out2 = torch.sigmoid(self.fc_lin4(out))

        return out1, out2

    def save(self, file_path):
        torch.save(self.state_dict(), file_path)


class PcNet_chemBERTa(nn.Module, ABC):
    def __init__(self, input_size_prot=1024, input_size_comp=768, hidden_size_prot=32, hidden_size_comp=24):
        super(PcNet_chemBERTa, self).__init__()
        self.fc_prot = nn.Linear(input_size_prot, hidden_size_prot)
        self.fc_comp = nn.Linear(input_size_comp, hidden_size_comp)
        self.fc_lin1 = nn.Linear(hidden_size_prot+hidden_size_comp, 1024)
        self.fc_drop1 = nn.Dropout(0.1)
        self.fc_lin2 = nn.Linear(1024, 1024)
        self.fc_drop2 = nn.Dropout(0.1)
        self.fc_lin3 = nn.Linear(1024, 512)
        self.fc_lin4 = nn.Linear(512, 1)

    def forward(self, x):
        out_prot = f.relu(self.fc_prot(x[0]))
        out_comp = f.relu(self.fc_comp(x[1]))
        out = torch.cat((out_prot, out_comp), dim=1)
        out = self.fc_drop1(f.relu(self.fc_lin1(out)))
        out = self.fc_drop2(f.relu(self.fc_lin2(out)))
        out = f.relu(self.fc_lin3(out))
        out1 = f.relu(self.fc_lin4(out))
        out2 = torch.sigmoid(self.fc_lin4(out))

        return out1, out2

    def save(self, file_path):
        torch.save(self.state_dict(), file_path)
        
        
class PcNet_RDKit(nn.Module):
    def __init__(self, input_size_prot=1024, input_size_comp=300, hidden_size_prot=32, hidden_size_comp=25):
        super(PcNet_RDKit, self).__init__()
        self.fc_prot = nn.Linear(input_size_prot, hidden_size_prot)
        self.fc_comp = nn.Linear(input_size_comp, hidden_size_comp)
        self.fc_lin1 = nn.Linear(hidden_size_prot+hidden_size_comp, 1024)
        self.fc_drop1 = nn.Dropout(0.1)
        self.fc_lin2 = nn.Linear(1024, 1024)
        self.fc_drop2 = nn.Dropout(0.1)
        self.fc_lin3 = nn.Linear(1024, 512)
        self.fc_lin4 = nn.Linear(512, 1)

    def forward(self, x):
        out_prot = f.relu(self.fc_prot(x[0]))
        out_comp = f.relu(self.fc_comp(x[1]))
        out = torch.cat((out_prot, out_comp), dim=1)
        out = self.fc_drop1(f.relu(self.fc_lin1(out)))
        out = self.fc_drop2(f.relu(self.fc_lin2(out)))
        out = f.relu(self.fc_lin3(out))
        out1 = f.relu(self.fc_lin4(out))
        out2 = torch.sigmoid(self.fc_lin4(out))

        return out1, out2

    def save(self, file_path):
        torch.save(self.state_dict(), file_path)


class EmbeddingReducingNN(nn.Module):
    def __init__(self, input_size_protein_embedding=1024, input_size_compund_embedding=196):
        super(EmbeddingReducingNN, self).__init__()

        self.fully_connected_protein_layer_1 = nn.Linear(input_size_protein_embedding,
                                                         round(input_size_protein_embedding / 2))
        self.fully_connected_protein_layer_2 = nn.Linear(round(input_size_protein_embedding / 2),
                                                         round(input_size_protein_embedding / 4))

        self.fully_connected_compound_layer_1 = nn.Linear(input_size_compund_embedding,
                                                          round(input_size_compund_embedding / 2))
        self.fully_connected_compound_layer_2 = nn.Linear(round(input_size_compund_embedding / 2),
                                                          round(input_size_compund_embedding / 4))

        self.fully_connected_combined_layer_1 = nn.Linear(round(input_size_protein_embedding / 4) +
                                                          round(input_size_compund_embedding / 4),
                                                          round(input_size_protein_embedding / 4) +
                                                          round(input_size_compund_embedding / 4)
                                                          )
        self.dropout_layer_1 = nn.Dropout(0.5)
        self.fully_connected_combined_layer_2 = nn.Linear(round(input_size_protein_embedding / 4) +
                                                          round(input_size_compund_embedding / 4),
                                                          round((round(input_size_protein_embedding / 4) +
                                                                 round(input_size_compund_embedding / 4)) / 2)
                                                          )
        self.dropout_layer_2 = nn.Dropout(0.5)
        self.fully_connected_combined_layer_3 = nn.Linear(round((round(input_size_protein_embedding / 4) +
                                                                 round(input_size_compund_embedding / 4)) / 2),
                                                          round((round(input_size_protein_embedding / 4) +
                                                                 round(input_size_compund_embedding / 4)) / 4)
                                                          )
        self.output_layer = nn.Linear(round((round(input_size_protein_embedding / 4) +
                                             round(input_size_compund_embedding / 4)) / 4), 1)

    def forward(self, input):
        reduced_protein = f.leaky_relu(self.fully_connected_protein_layer_1(input[0]))
        reduced_protein = f.leaky_relu(self.fully_connected_protein_layer_2(reduced_protein))

        reduced_compound = f.leaky_relu(self.fully_connected_compound_layer_1(input[1]))
        reduced_compound = f.leaky_relu(self.fully_connected_compound_layer_2(reduced_compound))

        combined_representation = torch.cat((reduced_protein, reduced_compound), dim=1)

        result = self.dropout_layer_1(f.leaky_relu(self.fully_connected_combined_layer_1(combined_representation)))
        result = self.dropout_layer_2(f.leaky_relu(self.fully_connected_combined_layer_2(result)))

        result = f.leaky_relu(self.fully_connected_combined_layer_3(result))

        out1 = f.relu(self.output_layer(result))
        out2 = torch.sigmoid(self.output_layer(result))

        return out1, out2

    def save(self, file_path):
        torch.save(self.state_dict(), file_path)
