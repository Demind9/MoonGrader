import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_add_pool, GATv2Conv, global_mean_pool

# --- GNN Version 2 ---

EMBED_SIZE  = 32
L0_SIZE     = EMBED_SIZE * 4
L1_SIZE     = EMBED_SIZE * 4 * 4
L2_SIZE     = EMBED_SIZE * 4 * 4 * 4
L3_SIZE     = EMBED_SIZE * 4 * 4 * 4 * 2

class GNNClassifier(torch.nn.Module):
    def __init__(self, num_classes, global_graph):
        super(GNNClassifier, self).__init__()
        
        self.global_graph = global_graph
        self.num_classes  = num_classes
        self.epsilon = 1e-8
        
        self.LNorm0 = nn.LayerNorm(L0_SIZE)
        self.LNorm1 = nn.LayerNorm(L1_SIZE)
        self.LNorm2 = nn.LayerNorm(L2_SIZE)
        self.LNorm3 = nn.LayerNorm(L3_SIZE)
        
        # Embedding and linear layers to transform input to reasonable size
        # Embedding layers for categorical inputs because it is more efficient
        # Linear layers for continuous because it captures more nuance
        
        #self.coordinate_embedding          = torch.nn.Linear(2, EMBED_SIZE) # Relu optional here (try with and without)
        ### THESE ARE TO BECOME LINEAR OR CONVOLUTIONAL BECAUSE OF CONTINUOUS NATURE OF INFO
        ### LIKE X COORD 1 IS LESS DIFFERENT FROM 2 THAN 10
        self.coordinate_embedding_x        = torch.nn.Embedding(11, EMBED_SIZE)
        self.coordinate_embedding_y        = torch.nn.Embedding(18, EMBED_SIZE)
        self.coordinate_smoosh             = torch.nn.Linear(EMBED_SIZE*2, EMBED_SIZE) # Relu after this
        
        self.position_embedding_1          = torch.nn.Embedding(140, EMBED_SIZE) # Maybe can just go directly to 16 with no Relu
        #self.position_embedding_2          = torch.nn.Linear(64, EMBED_SIZE)
        
        self.hold_type_one_hot_embedding   = torch.nn.Linear(5, EMBED_SIZE)
        
        self.orientation_one_hot_embedding = torch.nn.Linear(8, EMBED_SIZE)
        
        self.feature_mix                   = torch.nn.Linear(L0_SIZE, L0_SIZE)

        self.edge_weights_1                = torch.nn.Linear(EMBED_SIZE*6, L1_SIZE)
        self.edge_weights_2                = torch.nn.Linear(L1_SIZE, L0_SIZE)
        #(2073x128 and 32x512)

        # Embeddings for edge_weights / edge_attr / edge costs so that the network can learn which features are most
        # important when calculating the cost or closeness of an edge
        # Can be calculated as a Linear combination of everything but the specific position embeddings
        # THIS SHOULD BE APPLIED TO THE CONCATENATION OF BOTH NODES INVOLVED IN THE EDGE
        ### ALSO, NEED TO TAKE INVERSE BEFORE FEEDING TO EDGE WEIGHTS BECAUSE LARGE EDGE WEIGHTS ARE USUALLY MORE IMPORTANT
        ### MAYBE ALSO CONSIDER MAKING SOME THINGS SQUARED OR FOLLOW SOME SORTA LINEAR REGRESSION TYPE OF MULTIPLICATION
        #self.edge_weights_1 = F.relu(torch.nn.Linear(96, 32))
        # relu parts needs to be in forward pass
        #self.edge_weights_2 = 1.0 / (torch.nn.Linear(32, 1) + 1e-8)
        
        # Convolutions to groups of nodes
        self.conv1 = GATv2Conv(L0_SIZE, L1_SIZE, add_self_loops = False, edge_dim = L0_SIZE, dropout = 0.2)
        self.conv2 = GATv2Conv(L1_SIZE, L2_SIZE, add_self_loops = False, edge_dim = L0_SIZE, dropout = 0.2)
        self.conv3 = GATv2Conv(L2_SIZE, L3_SIZE, add_self_loops = False, edge_dim = L0_SIZE, dropout = 0.2)
        self.skip01 = torch.nn.Linear(L0_SIZE, L1_SIZE)
        self.skip02 = torch.nn.Linear(L0_SIZE, L2_SIZE)
        self.skip03 = torch.nn.Linear(L0_SIZE, L3_SIZE)
        self.skip12 = torch.nn.Linear(L1_SIZE, L2_SIZE)
        self.skip13 = torch.nn.Linear(L1_SIZE, L3_SIZE)
        self.skip23 = torch.nn.Linear(L2_SIZE, L3_SIZE)
        
        # Final layer to output logits for 11 classes.
        self.fc1 = torch.nn.Linear(L3_SIZE, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)
    
    def forward(self, climbs, drop_coef = 0):
        # Climbs come in batch in accordance w Torch-Geometric's DataLoader
        # Essentially will be 1 long vector of nodes with each node's specific info within it (and this will be of same length)
        # Edge weights can be directly calculated from the local edge indexes, by feeding the features from the 2 nodes from each 
        # edge and putting weights back in the same order
        
        # A sample batch shape
        #print(climbs)
        #DataBatch(x=[216, 213], edge_index=[2, 887], edge_attr=[887], y=[32], batch=[216], ptr=[33])
        
        
        climb_node_features = climbs.x
        climb_edge_indexes  = climbs.edge_index
        #drop_coef           = drop_sched.get_coef()
        #climb_dists         = climbs.edge_attr

        # coords            = climb_node_features[:, :2]                  # shape: [xB, 2]
        # coords            = self.coordinate_embedding(coords)
        # coords            = F.relu(coords)
        coords_y          = climb_node_features[:, 0].to(torch.long)    # shape: [xB, 1]
        coords_x          = climb_node_features[:, 1].to(torch.long)
        coords_y          = self.coordinate_embedding_y(coords_y)
        coords_x          = self.coordinate_embedding_x(coords_x)
        coords            = self.coordinate_smoosh(torch.cat([coords_y, coords_x], dim=1))
        coords            = F.leaky_relu(coords)

        positions         = climb_node_features[:, 2]   # shape: [xB, 1]
        positions         = positions.to(torch.long)
        positions         = self.position_embedding_1(positions)
        positions         = F.leaky_relu(positions)
        #positions         = self.position_embedding_2(positions)
        #positions         = positions.sum(dim = 1)
        
        hold_types        = climb_node_features[:, 3:8] # shape: [xB, 5]
        hold_types_sum    = hold_types.sum(dim=1, keepdim=True)
        hold_types        = self.hold_type_one_hot_embedding(hold_types)
        #hold_types        = hold_types.sum(dim=1) / hold_types_sum
        hold_types        = hold_types / (hold_types_sum + self.epsilon)
        
        orientations      = climb_node_features[:, 8:16]  # shape: [xB, 8]
        orientations_sum  = orientations.sum(dim=1, keepdim=True)
        orientations      = self.orientation_one_hot_embedding(orientations)
        #orientations      = orientations.sum(dim=1) / orientations_sum
        orientations      = orientations / (orientations_sum + self.epsilon)

        
        # [B, N, 213] -> [B, N, 64]
        x = torch.cat([coords, orientations, hold_types, positions], dim=1)
        x = self.feature_mix(x)
        x = self.LNorm0(x)
        x = F.relu(x)
        x = F.dropout(x, p = 0.4 * drop_coef)


        a = climb_edge_indexes[0]
        b = climb_edge_indexes[1]
        weights = torch.cat([coords[a], orientations[a], hold_types[a], coords[b], orientations[b], hold_types[b]], dim=1)
        weights = self.edge_weights_1(weights)
        weights = self.LNorm1(weights)
        weights = F.relu(weights)
        #weights = F.dropout(weights, p = 0.3 * drop_coef)
        weights = self.edge_weights_2(weights)
        weights = self.LNorm0(weights)
        weights = F.leaky_relu(weights)
        weights = F.dropout(weights, p = 0.2 * drop_coef)
        
        ### MultiheadAttention instead?

        ### RECALCULATE WEIGHTS ACROSS CONVOLUTIONAL LAYERS???
        ### Like have them change with each layer according to a transformer linear layer or something?
        ### Another possibility is to have weights feed in the same, but be of bigger dimension, so that there are more items for each layer to pay specific attention to
        
        # Apply GCN layers with ReLU activations.
        x1 = self.conv1(x , climb_edge_indexes, edge_attr = weights)
        x1 = self.LNorm1(x1)
        x1 = F.leaky_relu(x1, negative_slope=0.01)
        x1 = F.dropout(x1, p = 0.2 * drop_coef)
        x2 = self.conv2(x1, climb_edge_indexes, edge_attr = weights)
        x2 = self.LNorm2(x2)
        x2 = F.leaky_relu(x2, negative_slope=0.01)
        x2 = F.dropout(x2, p = 0.2 * drop_coef)
        x3 = self.conv3(x2, climb_edge_indexes, edge_attr = weights)
        x3 = self.LNorm3(x3)
        x3 = F.leaky_relu(x3, negative_slope=0.01)
        x3 = F.dropout(x3, p = 0.2 * drop_coef)

        xF = x3 + self.skip23(x2) + self.skip13(x1)
        # + self.skip23(x2)
        
        # Mean pooling to avoid trap of just predicting most common category and for normalization of climbs with more nodes
        #xF = global_mean_pool(xF, climbs.batch)
        # Sum pooling can be useful when trying to implement cumulative difficulty
        xF = global_add_pool(xF, climbs.batch)
        xF = F.dropout(xF, p = 0.5 * drop_coef)
        
        # Fully connected layer to get logits for each class.
        xF = self.fc1(xF)
        xF = F.relu(xF)
        logits = self.fc2(xF)
        # Log probabilities for KLDivLoss with soft target labels.
        # log_probs = F.log_softmax(logits, dim=1)
        
        return logits


# Basic dropout scheduler. get_coef() returns a value by which to multiply the dropout values within the model
class DropoutScheduler:
    def __init__(self, max_epochs, start_coef = 1, end_coef = 0.1, alpha = 1, beta = 1, off = False):
        self.max_epochs = max_epochs
        self.start_coef = start_coef
        self.end_coef   = end_coef
        self.alpha      = alpha
        self.beta       = beta
        self.off        = off
        self.epoch      = 0
        
    def step(self):
        self.epoch += 1
        
    def get_coef(self):
        return 1 if self.off == True else self.beta * (self.start_coef - (self.start_coef - self.end_coef) * (self.epoch**self.alpha / self.max_epochs**self.alpha))




# To instantiate the model.
# model = GNNClassifier(num_classes=11, global_graph=data)
# model(X_batch, drop_sched)

# To instantiate the dropout scheduler.
# drop_sched = DropoutScheduler(max_epochs, start_coef=_, end_coef=_)
# drop_sched.step()
# drop_sched.get_coef()

# to print the parameters
# for name, param in model.named_parameters():
#     print(f"Parameter: {name}")
#     print(f" - Shape: {param.shape}")
#     print(f" - Requires grad: {param.requires_grad}")
#     # You can print the entire tensor or summary statistics if it's large
#     print(f" - Values (mean): {param.data.mean().item():.4f}")
#     print(f" - Values (std): {param.data.std().item():.4f}\n")