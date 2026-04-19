import torch
s = torch.load('graph_data/snapshots_train.pt', weights_only=False)

# Check 1: snap_time_ns stored
print(hasattr(s[0], 'snap_time_ns'))          # must be True
print(s[0]['snap_time_ns'])                    # should be a tensor with one int64

# Check 2: reverse airport→flight edges exist
snap = next(x for x in s if x['flight'].num_nodes > 0)
print(hasattr(snap[('airport','departs_to','flight')], 'edge_index'))   # must be True
print(snap[('airport','departs_to','flight')].edge_index.shape)         # (2, ~1700)

# Check 3: tail_id on flight nodes
print(hasattr(snap['flight'], 'tail_id'))      # must be True
print(snap['flight'].tail_id[:5])              # should be int64 values in [1, 6788]