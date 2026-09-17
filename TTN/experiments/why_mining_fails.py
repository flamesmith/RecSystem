import sys
sys.path.insert(0, "/private/tmp/claude-501/-Users-lazr-PycharmProjects-RecSystem/4d057b13-3828-4b13-bed8-0ded1333e1f6/scratchpad")
exec(open(sys.argv[1] + "/collapse.py").read().split("rng = np.random.default_rng(0)")[0])
rng = np.random.default_rng(0)
POOL = 256
_order = np.argsort(node_of_item, kind="stable")
_starts = np.searchsorted(node_of_item[_order], np.arange(node_of_item.max()+2), side="left")
order_t = torch.tensor(_order, device=DEV); starts_t = torch.tensor(_starts, device=DEV)
def uniform_in_node(nid, n):
    lo, hi = starts_t[nid].unsqueeze(1), starts_t[nid+1].unsqueeze(1)
    size = (hi-lo).clamp_min(1)
    off = (torch.rand(len(nid), n, device=DEV)*size).long().clamp(max=size-1)
    return order_t[(lo+off).clamp(max=len(order_t)-1)]

# popularity percentile of an item inside its own node, from TRAIN counts
cnt = pairs_train.groupby("target_idx").size()
pop = np.zeros(len(node_of_item)); pop[cnt.index.to_numpy()] = cnt.to_numpy()
pct = np.zeros(len(node_of_item))
for nd in np.unique(node_of_item):
    m = np.flatnonzero(node_of_item == nd)
    if len(m) > 1:
        pct[m] = pd.Series(pop[m]).rank(pct=True).to_numpy()

# is the item a recorded target of ANY query, and of any query in this node?
any_target = np.zeros(len(node_of_item), bool)
any_target[np.union1d(pairs_train.target_idx.to_numpy(), pairs_test.target_idx.to_numpy())] = True
test_pairs = set(map(tuple, pairs_test[["query_idx","target_idx"]].to_numpy().tolist()))

with torch.no_grad():
    b = pairs_train.iloc[rng.choice(len(pairs_train), 4096, replace=False)]
    qi = torch.tensor(b.query_idx.astype(int).to_numpy(), device=DEV)
    ni = torch.tensor(b.target_node_id.astype(int).to_numpy(), device=DEV)
    poolv = uniform_in_node(ni, POOL)
    qv = model.query(qi, ni)
    s = torch.einsum("cd,cpd->cp", qv, cand[poolv])
    s = s.masked_fill(poolv == qi.unsqueeze(1), -1e4)
    mined = torch.gather(poolv, 1, torch.topk(s, 4, 1).indices).cpu().numpy()
    unif = uniform_in_node(ni, 4).cpu().numpy()
    tgt = b.target_idx.astype(int).to_numpy()

qn = b.query_idx.astype(int).to_numpy()
for name, negs in (("MINED (top-4 of 256)", mined), ("UNIFORM (4 in-node)", unif)):
    f = negs.ravel()
    fn_test = np.mean([(qn[i], int(x)) in test_pairs for i in range(len(negs)) for x in negs[i]])
    print(f"{name}")
    print(f"   mean popularity percentile within node : {pct[f].mean():.3f}")
    print(f"   share in the node's top-10 by train count: "
          f"{np.mean([pct[x] > 1 - 10/max((node_of_item==node_of_item[x]).sum(),1) for x in f[:2000]]):.3f}")
    print(f"   share that are a recorded target somewhere: {any_target[f].mean():.3f}")
    print(f"   share that are a TEST complement of this very query: {fn_test:.4f}")
print(f"\nfor reference, the true targets' mean popularity percentile: {pct[tgt].mean():.3f}")
