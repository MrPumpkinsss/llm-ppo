"""V7: Autoregressive GNN-PPO with Positional Encoding + Pipeline-Aware Selection.

Combines V6's GNN backbone with V5's autoregressive selection:
- After each device selection, update node features with positional encoding
- Re-run GNN at each step to get fresh embeddings informed by pipeline state
- Fully observable MDP: no GRU needed, GNN + dynamic features carry full state
- Pointer attention conditioned on graph + layer context + global dynamics
- Per-node dynamic features: is_selected, pipeline_position, bw_to_pipeline_tail
- Global dynamic features: step progress, cumulative compute ratio
- STOP action allows early termination (skip suboptimal devices)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.shared import MAX_DEVICES, MAX_LAYERS
from baselines import min_sum_tpot_dp
from environment import compute_simple_tpot

STOP_ACTION = MAX_DEVICES  # action index for STOP


# ===================== Observation Builder =====================

def build_v7_graph_observation(devices, layers, tensor_size, num_layers, num_devices):
    """Build V7 graph inputs with extended node features (11-dim).

    Node features per device (11-dim):
        [0] compute_power           (static)
        [1] mean_bandwidth / 5.0    (static)
        [2] min_bandwidth / 5.0     (static)
        [3] max_bandwidth / 5.0     (static)
        [4] workload_hint            (static: total_layer_cost / compute_power / 100)
        [5] num_layers / MAX_LAYERS (static)
        [6] num_devices / MAX_DEVICES (static)
        [7] is_valid                (static: 1.0 for real devices)
        [8] is_selected             (dynamic: 0/1)
        [9] pipeline_position_norm  (dynamic: 0 if not selected, else (pos+1)/nd)
        [10] bw_to_pipeline_tail    (dynamic: bandwidth to last selected device / 5.0)

    Returns:
        node_feats: (MAX_DEVICES, 11)
        edge_feats: (MAX_DEVICES, MAX_DEVICES, 3)
        adj_mask: (MAX_DEVICES, MAX_DEVICES) bool
        layer_costs: (MAX_LAYERS,) normalized
    """
    nd = num_devices
    nl = num_layers

    # --- Node features (11-dim) ---
    node_feats = np.zeros((MAX_DEVICES, 11), dtype=np.float32)
    total_layer_cost = float(layers.compute_costs[:nl].sum())

    for i in range(nd):
        node_feats[i, 0] = devices.compute_power[i]
        if nd > 1:
            bws = [devices.bandwidth[i, j] for j in range(nd) if j != i]
            node_feats[i, 1] = np.mean(bws) / 5.0
            node_feats[i, 2] = np.min(bws) / 5.0
            node_feats[i, 3] = np.max(bws) / 5.0
        node_feats[i, 4] = total_layer_cost / (devices.compute_power[i] + 1e-8) / 100.0
        node_feats[i, 5] = nl / MAX_LAYERS
        node_feats[i, 6] = nd / MAX_DEVICES
        node_feats[i, 7] = 1.0  # is_valid
        # [8], [9], [10] are dynamic, initially 0

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
                    edge_feats[i, j, 0] = bw / 5.0
                    edge_feats[i, j, 1] = 1.0 / (bw + 1e-8) * 0.5
                    edge_feats[i, j, 2] = 1.0 if bw > median_bw else 0.0
                    adj_mask[i, j] = True

    # --- Layer costs ---
    layer_costs = np.zeros(MAX_LAYERS, dtype=np.float32)
    layer_costs[:nl] = layers.compute_costs[:nl].copy()
    max_cost = layer_costs[:nl].max() if nl > 0 else 1.0
    if max_cost > 0:
        layer_costs[:nl] /= max_cost

    return node_feats, edge_feats, adj_mask, layer_costs


def update_dynamic_features(node_feats, devices, selected, num_devices):
    """Update dynamic node features [8:11] based on current selection state.

    Args:
        node_feats: (MAX_DEVICES, 11) - will be modified in-place
        devices: DeviceCluster
        selected: list of device indices selected so far (in order)
        num_devices: int
    """
    last_selected = selected[-1] if selected else None

    for d in range(num_devices):
        # is_selected
        node_feats[d, 8] = 1.0 if d in selected else 0.0

        # pipeline_position_norm
        if d in selected:
            pos = selected.index(d)
            node_feats[d, 9] = (pos + 1) / num_devices
        else:
            node_feats[d, 9] = 0.0

        # bw_to_pipeline_tail: bandwidth to the most recently selected device
        if last_selected is not None and d != last_selected:
            node_feats[d, 10] = devices.bandwidth[d, last_selected] / 5.0
        else:
            node_feats[d, 10] = 0.0


# ===================== GNN Layers =====================

class ECGCLayer(nn.Module):
    """Edge-Conditioned Graph Convolution Layer.

    Uses Mean + Max dual aggregation to capture both average and extreme signals.
    Max aggregation captures outlier devices (very fast/slow) that strongly
    influence the optimal partitioning decision.
    """

    def __init__(self, hidden_dim=128, edge_dim=64):
        super().__init__()
        self.edge_gate = nn.Sequential(
            nn.Linear(hidden_dim + 3, edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, hidden_dim),
            nn.Sigmoid(),
        )
        # update input: [self | agg_mean | agg_max] = hidden_dim * 3
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_feats, adj_mask):
        B, N, D = h.shape
        h_j = h.unsqueeze(2).expand(B, N, N, D)
        gate_input = torch.cat([h_j, edge_feats], dim=-1)
        gates = self.edge_gate(gate_input)
        messages = gates * h_j
        mask = adj_mask.float().unsqueeze(-1)
        messages = messages * mask

        # Mean aggregation (overall tendency)
        neighbor_count = mask.sum(dim=2).clamp(min=1.0)
        agg_mean = messages.sum(dim=2) / neighbor_count

        # Max aggregation (captures extreme / outlier devices)
        messages_for_max = messages.masked_fill(~adj_mask.bool().unsqueeze(-1), float('-inf'))
        agg_max = messages_for_max.max(dim=2)[0]
        # Handle isolated nodes (all -inf) → zero
        agg_max = torch.where(torch.isinf(agg_max), torch.zeros_like(agg_max), agg_max)

        update_input = torch.cat([h, agg_mean, agg_max], dim=-1)
        h_new = self.layer_norm(h + self.update(update_input))
        return h_new


# ===================== Network =====================

class PPOv7Network(nn.Module):
    """Autoregressive GNN-PPO with positional encoding (no GRU).

    Fully observable MDP: GNN re-runs each step with updated node features
    (is_selected, pipeline_position, bw_to_tail), so no recurrent state needed.

    At each selection step:
    1. Node features updated with is_selected, pipeline_position, bw_to_tail
    2. GNN processes the updated graph → fresh embeddings per step
    3. Pointer attention: query = [graph_embed, layer_embed, global_dyn]
    4. STOP logit from [graph_embed, global_dyn]
    5. Value from [graph_embed, layer_embed, global_dyn]
    """

    def __init__(self, hidden_dim=256, num_gnn_layers=3, max_devices=MAX_DEVICES):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_devices = max_devices

        # Node feature projection: 11-dim → hidden_dim
        self.node_proj = nn.Linear(11, hidden_dim)

        # GNN layers (re-run each step)
        self.gnn_layers = nn.ModuleList([
            ECGCLayer(hidden_dim, edge_dim=64) for _ in range(num_gnn_layers)
        ])

        # Layer context encoder
        self.layer_conv = nn.Conv1d(1, 32, kernel_size=4, stride=2)
        self.layer_proj = nn.Linear(32 * 31, hidden_dim)

        # Policy head: pointer attention
        # query = [graph_embed(D) | layer_embed(D) | global_dyn(2)]
        self.query_proj = nn.Linear(hidden_dim * 2 + 2, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)

        # STOP action head
        # input = [graph_embed(D) | global_dyn(2)]
        self.stop_head = nn.Sequential(
            nn.Linear(hidden_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Value head
        # input = [graph_embed(D) | layer_embed(D) | global_dyn(2)]
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward_step(self, node_feats, edge_feats, adj_mask, layer_costs, global_dynamic):
        """Single-step forward pass (batched).

        Args:
            node_feats: (B, N, 11) with dynamic features updated
            edge_feats: (B, N, N, 3)
            adj_mask: (B, N, N) bool
            layer_costs: (B, MAX_LAYERS)
            global_dynamic: (B, 2) = [step/MAX_DEVICES, cum_compute/total_compute]

        Returns:
            all_logits: (B, N+1) = [device_0..N-1, STOP]
            value: (B, 1)
            node_embeds: (B, N, D)
        """
        # Layer embedding
        layer_embed = F.relu(self.layer_conv(layer_costs.unsqueeze(1)))
        layer_embed = self.layer_proj(layer_embed.flatten(1))  # (B, D)

        # GNN
        h = self.node_proj(node_feats)
        for gnn in self.gnn_layers:
            h = gnn(h, edge_feats, adj_mask)  # (B, N, D)

        # Graph-level embedding (compute-power weighted mean over valid nodes)
        is_valid = node_feats[:, :, 7]  # (B, N)
        weights = (node_feats[:, :, 0] * is_valid).unsqueeze(-1)  # (B, N, 1)
        graph_embed = (weights * h).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-8)  # (B, D)

        # Policy: pointer attention
        query_input = torch.cat([graph_embed, layer_embed, global_dynamic], dim=-1)
        query = self.query_proj(query_input)  # (B, D)
        keys = self.key_proj(h)  # (B, N, D)
        device_scores = torch.bmm(keys, query.unsqueeze(2)).squeeze(2) / (self.hidden_dim ** 0.5)  # (B, N)

        # STOP logit
        stop_input = torch.cat([graph_embed, global_dynamic], dim=-1)
        stop_logit = self.stop_head(stop_input)  # (B, 1)

        # Combine: [device_scores | stop_logit]
        all_logits = torch.cat([device_scores, stop_logit], dim=1)  # (B, N+1)

        # Value
        value = self.value_head(query_input)  # (B, 1)

        return all_logits, value, h


# ===================== Episode Generation =====================

def ppo_v7_generate_episode(
    network, devices, layers, tensor_size,
    num_layers, num_devices, torch_device,
    deterministic=False, temperature=1.0,
):
    """Generate one autoregressive episode with GNN-PPO.

    At each step:
    1. Update dynamic node features (is_selected, pipeline_pos, bw_to_tail)
    2. Compute global dynamic features (step, cumulative compute)
    3. Forward pass → device scores + STOP logit
    4. Mask invalid actions, sample or greedy select
    5. If STOP: end episode

    Returns:
        step_data: list of tuples per step
        ordering: device ordering
        partition: DP partition
        tpot: TPOT
    """
    network.eval()

    # Build initial graph observation (dynamic features all zeros)
    node_feats, edge_feats, adj_mask, layer_costs = build_v7_graph_observation(
        devices, layers, tensor_size, num_layers, num_devices
    )

    total_compute = float(devices.compute_power[:num_devices].sum())

    selected = []
    step_data = []

    for step in range(num_devices + 1):
        remaining = [d for d in range(num_devices) if d not in selected]
        if not remaining:
            break

        # Update dynamic node features
        node_feats_updated = node_feats.copy()
        update_dynamic_features(node_feats_updated, devices, selected, num_devices)

        # Global dynamic features
        cumulative_compute = sum(devices.compute_power[d] for d in selected) / (total_compute + 1e-8)
        global_dynamic = np.array([step / MAX_DEVICES, cumulative_compute], dtype=np.float32)

        # Convert to tensors (B=1)
        node_t = torch.FloatTensor(node_feats_updated).unsqueeze(0).to(torch_device)
        edge_t = torch.FloatTensor(edge_feats).unsqueeze(0).to(torch_device)
        adj_t = torch.BoolTensor(adj_mask).unsqueeze(0).to(torch_device)
        layer_t = torch.FloatTensor(layer_costs).unsqueeze(0).to(torch_device)
        gd_t = torch.FloatTensor(global_dynamic).unsqueeze(0).to(torch_device)

        # Forward pass
        with torch.no_grad():
            all_logits, value, _ = network.forward_step(
                node_t, edge_t, adj_t, layer_t, gd_t
            )

        # Build action mask: [device_0..MAX_DEVICES-1, STOP]
        action_mask = np.zeros(STOP_ACTION + 1, dtype=bool)
        for d in remaining:
            action_mask[d] = True
        action_mask[STOP_ACTION] = len(selected) > 0  # can stop only if >= 1 device selected

        # Apply mask to logits
        mask_t = torch.BoolTensor(action_mask).unsqueeze(0).to(torch_device)
        masked_logits = all_logits.clone()
        masked_logits[~mask_t] = float('-inf')

        if temperature != 1.0:
            masked_logits = masked_logits / temperature

        probs = F.softmax(masked_logits, dim=-1)

        if deterministic:
            action = probs.argmax(dim=-1).item()
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()

        log_prob = torch.log(probs[0, action].clamp(min=1e-8)).item()

        # Store transition: everything needed for PPO update replay
        step_data.append((
            node_feats_updated.copy(),   # [0] node_feats
            edge_feats.copy(),           # [1] edge_feats (static)
            adj_mask.copy(),             # [2] adj_mask (static)
            layer_costs.copy(),          # [3] layer_costs (static)
            global_dynamic.copy(),       # [4] global_dynamic (2,)
            action,                      # [5] action taken
            log_prob,                    # [6] old log_prob
            value.item(),                # [7] old value
            action_mask.copy(),          # [8] action mask
        ))

        # Check STOP
        if action == STOP_ACTION:
            break

        # Select device
        selected.append(action)

    # Compute partition via DP
    ordering = selected if selected else [0]
    partition = min_sum_tpot_dp(num_layers, ordering, devices, layers, tensor_size)
    tpot = compute_simple_tpot(partition, devices, layers, tensor_size)

    return step_data, ordering, partition, tpot


# ===================== Inference =====================

def ppo_v7_inference(
    network, devices, layers, tensor_size,
    num_layers, num_devices, torch_device,
    num_candidates=2,
):
    """Inference: best-of-N stochastic candidates, or single deterministic pass."""
    network.eval()
    best_partition = None
    best_tpot = float('inf')

    for _ in range(num_candidates):
        deterministic = (num_candidates == 1)
        _, _, partition, tpot = ppo_v7_generate_episode(
            network, devices, layers, tensor_size,
            num_layers, num_devices, torch_device,
            deterministic=deterministic,
        )
        if tpot < best_tpot:
            best_tpot = tpot
            best_partition = partition

    return best_partition, best_tpot
