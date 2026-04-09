"""V6: GNN-Based PPO + Edge-Conditioned Graph Conv + min-max DP.

Devices form a complete graph with bandwidth as edge features.
GNN processes the topology, pointer attention produces device ordering,
DP finds optimal contiguous partition.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.shared import (
    MAX_DEVICES, MAX_LAYERS, build_observation,
)
from baselines import min_max_bottleneck_dp
from environment import compute_simple_tpot


# ============= Graph Construction =============

def build_graph_observation(devices, layers, tensor_size, num_layers, num_devices):
    """Build GNN inputs from environment.

    Returns:
        node_feats: (MAX_DEVICES, 8) per-device features
        edge_feats: (MAX_DEVICES, MAX_DEVICES, 3) per-edge features
        adj_mask: (MAX_DEVICES, MAX_DEVICES) bool, True for valid edges
        layer_costs: (MAX_LAYERS,) normalized layer compute costs
    """
    nd = num_devices
    nl = num_layers

    # --- Node features (8-dim) ---
    node_feats = np.zeros((MAX_DEVICES, 8), dtype=np.float32)
    total_layer_cost = float(layers.compute_costs[:nl].sum())

    for i in range(nd):
        node_feats[i, 0] = devices.compute_power[i]                  # compute power
        if nd > 1:
            bws = [devices.bandwidth[i, j] for j in range(nd) if j != i]
            node_feats[i, 1] = np.mean(bws) / 5.0                    # mean bw
            node_feats[i, 2] = np.min(bws) / 5.0                     # min bw
            node_feats[i, 3] = np.max(bws) / 5.0                     # max bw
        node_feats[i, 4] = total_layer_cost / (devices.compute_power[i] + 1e-8) / 100.0  # workload hint
        node_feats[i, 5] = nl / MAX_LAYERS                           # num_layers norm
        node_feats[i, 6] = nd / MAX_DEVICES                          # num_devices norm
        node_feats[i, 7] = 1.0                                       # is_valid

    # --- Edge features (3-dim) ---
    edge_feats = np.zeros((MAX_DEVICES, MAX_DEVICES, 3), dtype=np.float32)
    adj_mask = np.zeros((MAX_DEVICES, MAX_DEVICES), dtype=bool)

    if nd > 1:
        all_bws = []
        for i in range(nd):
            for j in range(i + 1, nd):
                all_bws.append(devices.bandwidth[i, j])
        median_bw = np.median(all_bws) if all_bws else 2.5

        for i in range(nd):
            for j in range(nd):
                if i != j:
                    bw = devices.bandwidth[i, j]
                    edge_feats[i, j, 0] = bw / 5.0                   # bandwidth norm
                    edge_feats[i, j, 1] = 1.0 / (bw + 1e-8) * 0.5    # transfer cost proxy
                    edge_feats[i, j, 2] = 1.0 if bw > median_bw else 0.0  # fast link flag
                    adj_mask[i, j] = True

    # --- Layer costs ---
    layer_costs = np.zeros(MAX_LAYERS, dtype=np.float32)
    layer_costs[:nl] = layers.compute_costs[:nl].copy()
    max_cost = layer_costs[:nl].max() if nl > 0 else 1.0
    if max_cost > 0:
        layer_costs[:nl] /= max_cost

    return node_feats, edge_feats, adj_mask, layer_costs


# ============= GNN Layers =============

class ECGCLayer(nn.Module):
    """Edge-Conditioned Graph Convolution Layer.

    Messages are gated by edge features: sigmoid(MLP(concat(h_j, e_ij))) * h_j
    """

    def __init__(self, hidden_dim=128, edge_dim=64):
        super().__init__()
        # Edge gating: concat(h_j, e_ij) -> gate
        self.edge_gate = nn.Sequential(
            nn.Linear(hidden_dim + 3, edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, hidden_dim),
            nn.Sigmoid(),
        )
        # Node update: concat(h_i, aggregated_messages) -> h_i_new
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_feats, adj_mask):
        """
        Args:
            h: (B, N, D) node hidden states
            edge_feats: (B, N, N, 3) edge features
            adj_mask: (B, N, N) bool mask, True = valid edge
        Returns:
            h_new: (B, N, D) updated node states
        """
        B, N, D = h.shape

        # h_j repeated for all edges: (B, N, N, D)
        h_j = h.unsqueeze(2).expand(B, N, N, D)

        # Edge gating: sigmoid(MLP(concat(h_j, e_ij)))
        gate_input = torch.cat([h_j, edge_feats], dim=-1)        # (B, N, N, D+3)
        gates = self.edge_gate(gate_input)                        # (B, N, N, D)

        # Gated messages
        messages = gates * h_j                                    # (B, N, N, D)

        # Mask out invalid edges
        mask = adj_mask.float().unsqueeze(-1)                     # (B, N, N, 1)
        messages = messages * mask                                # (B, N, N, D)

        # Aggregate: sum over source nodes (dim=2)
        agg = messages.sum(dim=2)                                 # (B, N, D)

        # Update with residual + LayerNorm
        update_input = torch.cat([h, agg], dim=-1)               # (B, N, 2D)
        h_new = self.layer_norm(h + self.update(update_input))    # (B, N, D)

        return h_new


# ============= Full Network =============

class PPOv6Network(nn.Module):
    """GNN-based PPO network: device graph + layer context -> ordering scores."""

    def __init__(self, hidden_dim=256, num_gnn_layers=3, max_devices=MAX_DEVICES):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_devices = max_devices

        # Node feature projection
        self.node_proj = nn.Linear(8, hidden_dim)

        # GNN layers
        self.gnn_layers = nn.ModuleList([
            ECGCLayer(hidden_dim, edge_dim=64) for _ in range(num_gnn_layers)
        ])

        # Layer context encoder: Conv1d over layer costs
        self.layer_conv = nn.Conv1d(1, 32, kernel_size=4, stride=2)
        # Conv1d output: (B, 32, floor((64-4)/2)+1) = (B, 32, 31)
        self.layer_proj = nn.Linear(32 * 31, hidden_dim)

        # Policy head: pointer attention (query uses both layer + graph context)
        self.query_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_feats, edge_feats, adj_mask, layer_costs):
        """
        Args:
            node_feats: (B, MAX_DEVICES, 8)
            edge_feats: (B, MAX_DEVICES, MAX_DEVICES, 3)
            adj_mask: (B, MAX_DEVICES, MAX_DEVICES) bool
            layer_costs: (B, MAX_LAYERS)
        Returns:
            attention_scores: (B, MAX_DEVICES)
            value: (B, 1)
        """
        B = node_feats.shape[0]

        # 1. Project node features
        h = self.node_proj(node_feats)                            # (B, N, D)

        # 2. GNN message passing
        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h, edge_feats, adj_mask)               # (B, N, D)

        # 3. Layer context
        layer_embed = self.layer_conv(layer_costs.unsqueeze(1))   # (B, 32, 31)
        layer_embed = F.relu(layer_embed)
        layer_embed = self.layer_proj(layer_embed.flatten(1))     # (B, D)

        # 4. Graph-level embedding (shared by policy + value)
        weights = (node_feats[:, :, 0] * node_feats[:, :, 7])    # (B, N)
        weights_unsq = weights.unsqueeze(-1)                      # (B, N, 1)
        graph_embed = (weights_unsq * h).sum(dim=1)               # (B, D)
        graph_embed = graph_embed / (weights_unsq.sum(dim=1).clamp(min=1e-8))

        # 5. Pointer attention (policy) — query uses graph + layer context
        query = self.query_proj(torch.cat([graph_embed, layer_embed], dim=-1))  # (B, D)
        keys = self.key_proj(h)                                   # (B, N, D)
        scores = torch.bmm(keys, query.unsqueeze(2)).squeeze(2)  # (B, N)
        scores = scores / np.sqrt(self.hidden_dim)

        # 6. Value: graph + layer context
        value = self.value_head(torch.cat([graph_embed, layer_embed], dim=-1))

        return scores, value


# ============= Ordering Generation =============

def ppo_v6_generate_ordering(
    network, node_feats, edge_feats, adj_mask, layer_costs,
    num_devices, deterministic=False, temperature=1.0,
):
    """Generate device ordering via sequential masking (same pattern as V3).

    Returns:
        ordering_batch: list of device index lists
        log_prob_batch: tensor of total log probs
        attention_scores: raw scores
        value: (B, 1)
    """
    attention_scores, value = network(node_feats, edge_feats, adj_mask, layer_costs)

    B = node_feats.shape[0]
    ordering_batch = []
    log_prob_batch = []

    for b in range(B):
        scores = attention_scores[b].clone()
        mask = torch.zeros(network.max_devices, device=node_feats.device, dtype=torch.bool)
        mask[:num_devices] = True

        ordering = []
        total_log_prob = 0.0

        for step in range(num_devices):
            masked_scores = scores.clone()
            masked_scores[~mask] = float('-inf')

            probs = F.softmax(masked_scores / temperature, dim=0)

            if deterministic:
                action = probs.argmax().item()
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()

            log_p = torch.log(probs[action].clamp(min=1e-8)).item()
            total_log_prob += log_p

            ordering.append(action)
            mask[action] = False

        ordering_batch.append(ordering)
        log_prob_batch.append(total_log_prob)

    return ordering_batch, torch.FloatTensor(log_prob_batch).to(node_feats.device), attention_scores, value


# ============= Inference =============

def ppo_v6_inference(
    network, devices, layers, tensor_size,
    num_layers, num_devices, torch_device,
    num_candidates = 2,
):
    """Policy-guided sampling: best-of-N orderings via DP.

    Sample N device orderings from the policy, run DP on each,
    return the partition with lowest TPOT.
    """
    network.eval()

    node_feats, edge_feats, adj_mask, layer_costs = build_graph_observation(
        devices, layers, tensor_size, num_layers, num_devices
    )

    node_t = torch.FloatTensor(node_feats).unsqueeze(0).to(torch_device)
    edge_t = torch.FloatTensor(edge_feats).unsqueeze(0).to(torch_device)
    adj_t = torch.BoolTensor(adj_mask).unsqueeze(0).to(torch_device)
    layer_t = torch.FloatTensor(layer_costs).unsqueeze(0).to(torch_device)

    # 1st candidate: deterministic greedy
    with torch.no_grad():
        greedy_orderings, _, _, _ = ppo_v6_generate_ordering(
            network, node_t, edge_t, adj_t, layer_t,
            num_devices, deterministic=True
        )

    best_partition = min_max_bottleneck_dp(
        num_layers, greedy_orderings[0], devices, layers, tensor_size
    )
    best_tpot = compute_simple_tpot(best_partition, devices, layers, tensor_size)

    # Remaining candidates: stochastic sampling with temperature
    if num_candidates > 1:
        for _ in range(num_candidates - 1):
            with torch.no_grad():
                sampled_orderings, _, _, _ = ppo_v6_generate_ordering(
                    network, node_t, edge_t, adj_t, layer_t,
                    num_devices, deterministic=False, temperature=1.0
                )

            ordering = sampled_orderings[0]
            # Skip duplicate orderings
            if ordering == greedy_orderings[0]:
                continue
            partition = min_max_bottleneck_dp(num_layers, ordering, devices, layers, tensor_size)
            tpot = compute_simple_tpot(partition, devices, layers, tensor_size)

            if tpot < best_tpot:
                best_tpot = tpot
                best_partition = partition

    return best_partition, best_tpot
