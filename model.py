import torch
import torch.nn as nn
import random

class Vocab:
    def __init__(self, node_dict, nodeindex_dict, edge_dict, edge_decode_dict):
        self.node_dict = node_dict
        self.nodeindex_dict = nodeindex_dict    
        self.edge_dict = edge_dict
        self.edge_decode_dict = edge_decode_dict

    def __call__(self, x):
        if isinstance(x, list):
            return [self.__call__(_) for _ in x]
        else:
            return self.fetch(x)

    def fetch(self, x):
        s, t = x.split("->")
        return self.edge_dict[s][t] if s in self.edge_dict and t in self.edge_dict[s] else self.edge_dict["<unk>"]["<unk>"]

    @classmethod
    def from_node_dict(cls, dictname):
        nodeindex_dict = dict()
        edge_dict = dict()
        edge_decode_dict = dict()
        for s in dictname:
            nodeindex_dict[dictname[s]] = s
            edge_dict[s] = {}
            for t in dictname:
                edge_dict[s][t] = (dictname[s], dictname[t])
                edge_decode_dict[(dictname[s], dictname[t])] = "->".join([s, t])
        return cls(None, nodeindex_dict, edge_dict, edge_decode_dict)

    @classmethod
    def from_edge(cls, filename):
        edge_dict = dict()
        edge_dict["<unk>"] = {}
        edge_dict["<unk>"]["<unk>"] = (0, 0)
        edge_decode_dict = dict()
        with open(filename) as f:
            for line in f:
                s, t = line.strip().split("->")
                if s not in edge_dict:
                    i = len(edge_dict)
                    j = 0
                    edge_dict[s] = dict()
                else:
                    i = edge_dict[s][list(edge_dict[s].keys())[0]][0]
                    j = len(edge_dict[s])
                edge_dict[s][t] = (i, j)
                edge_decode_dict[(i, j)] = "->".join([s, t])
        return cls(None, edge_dict, edge_decode_dict)

    def get_neighbor_of_edge(self, key, k):
        s, t = key.split("->")
        _s = s if s in self.edge_dict else "<unk>"
        ret = ["->".join([_s, _t]) for _t in self.edge_dict[_s].keys() if _t != t]
        random.shuffle(ret)
        return ret[:k] if k != -1 else ret

    def get_neighbor_of_node(self, key, k):
        s = self.nodeindex_dict[key]
        ret = ["->".join([s, _t]) for _t in self.edge_dict[s].keys() if _t != s]
        random.shuffle(ret)
        return ret[:k] if k != -1 else ret
    
    def get_neighbor_of_edge_broadcast(self, key, edges, k=100):
        s, t = key.split("->")
        _ret = [_t for _t in self.edge_dict[s].keys() if _t != t]
        random.shuffle(_ret)
        ret = []
        for edge in edges:
            s, t = edge.split("->")
            ret += [["->".join([s, _t]) for _t in _ret[:k]]]
        return ret

    @staticmethod
    def to_path(tokens):
        path = []
        for left, right in zip(tokens[:-1], tokens[1:]):
            path.append("->".join([left, right]))
        return path

    def get_edge_of_node(self, key):
        return list(self.edge_dict[key].values())

    def decode(self, x):
        return self.edge_decode_dict[x]
    

class BraLM(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.network = nn.ParameterList()
        self.bias = nn.ParameterList()
        self.sigmoid = nn.GELU()
        self.positions = nn.Parameter(torch.ones(1, 512, 1))
        self.device = None

    def prepare_network(self, vocab):
        for s in vocab.edge_dict:
            self.network.append(nn.Parameter(torch.randn(len(vocab.edge_dict[s]), self.hidden_size, self.hidden_size).uniform_(-0.5, 0.5)))
            self.bias.append(nn.Parameter(torch.randn(len(vocab.edge_dict[s]), 1, self.hidden_size).uniform_(-0.5, 0.5)))

    def _network(self, x, y):
        return self.network[x][y]

    def to_device(self, device):
        self.network.to(device)
        self.positions.data = self.positions.data.to(device)
        self.device = device

    @staticmethod
    def _reshape12(x):
        return x.reshape(-1, x.size(-2), x.size(-1))

    def get_positional_encoding(self, seq_len, d_model):
        position = torch.arange(0, seq_len).reshape(-1, 1)
        div_term = 10000.0 ** (torch.arange(0, d_model, 2) / d_model)
        position_encoding = torch.zeros(seq_len, d_model)
        position_encoding[:, 0::2] = torch.sin(position * div_term)
        position_encoding[:, 1::2] = torch.cos(position * div_term)
        return position_encoding.unsqueeze(0).to(self.device)


    def get_initial_tensor(self, batch_size):
        energy_tensor = torch.ones(batch_size, 1, self.hidden_size) / self.hidden_size
        return energy_tensor.to(self.device)


    def decode(self, start, vocab, max_new_tokens=16, do_sample=False, temperature=1):
        ret = []
        pe = self.get_positional_encoding(512, self.hidden_size)
        for i, pair in enumerate(start):
            if i == 0:
                energy_tensor = self.get_initial_tensor(batch_size=1).squeeze(0)
            else:
                energy_tensor = (energy_cache * self.positions[:, :i, :].softmax(1)).sum(1, keepdim=True).squeeze(0)
            w = self._network(pair[0], pair[1]).to(self.device)
            b = self.bias[pair[0]][pair[1]].to(self.device)

            energy_tensor = self.sigmoid(energy_tensor.mm(w) + b + pe.squeeze(0)[i])
            if i == 0:
                energy_cache = energy_tensor
            else:
                energy_cache = torch.cat([energy_cache, energy_tensor], dim=0)
            ret += [pair]
        x = pair[1]
        prev_i = len(start)

        for i in range(max_new_tokens):
            candidates = vocab(vocab.get_neighbor_of_node(x, -1))
            all_w = torch.cat([self._network(z[0], z[1]).unsqueeze(0) for z in candidates], dim=0).to(self.device)
            all_b = torch.cat([self.bias[z[0]][z[1]].unsqueeze(0) for z in candidates], dim=0).to(self.device)

            curr_i = prev_i + i
            energy_tensor = (energy_cache * self.positions.squeeze(0)[:curr_i, :].softmax(0)).sum(0, keepdim=True)
            expand_energy_tensor = energy_tensor.unsqueeze(0).repeat(all_w.size(0), 1, 1)

            nxt_energy_tensor = self.sigmoid(expand_energy_tensor.bmm(all_w)+all_b+pe[:,i])

            energy = nxt_energy_tensor.norm(2, (-2,-1))

            probs = torch.softmax(energy, dim=-1)
            if temperature > 0:
                probs = probs / temperature
            if do_sample:
                index = torch.multinomial(probs, 1).item()
            else:
                index = probs.argmax(-1).item()

            y = candidates[index][-1]
            ret += [(x, y)]

            energy_tensor = nxt_energy_tensor[index, :, :]
            x = y

            energy_cache = torch.cat([energy_cache, energy_tensor], dim=0)

        return ret