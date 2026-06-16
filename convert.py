"""Copy a PyTorch state_dict into a JAX PyTree.

nn.Linear stores its weight as (out_features, in_features) and computes x @ W.T,
while the JAX model computes x @ W with W shaped (in, out), so every Linear
weight is transposed during the copy. Embeddings and norm gains copy directly.
lm_head is tied to tok_emb and is not copied separately; the JAX side uses
tok_emb.T for the output projection.
"""

import jax.numpy as jnp


def torch_to_jax(model):
    """model: model_torch.GPT (any device). Returns a model_jax params tree."""
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    cfg = model.cfg
    params = {
        "tok_emb": jnp.asarray(sd["tok_emb.weight"]),
        "ln_f_g": jnp.asarray(sd["ln_f.g"]),
        "blocks": [],
    }
    for i in range(cfg.n_layers):
        p = f"blocks.{i}."
        params["blocks"].append({
            "ln1_g": jnp.asarray(sd[p + "ln1.g"]),
            "wq": jnp.asarray(sd[p + "attn.wq.weight"].T),  # (out,in)->(in,out)
            "wk": jnp.asarray(sd[p + "attn.wk.weight"].T),
            "wv": jnp.asarray(sd[p + "attn.wv.weight"].T),
            "wo": jnp.asarray(sd[p + "attn.wo.weight"].T),
            "ln2_g": jnp.asarray(sd[p + "ln2.g"]),
            "fc": jnp.asarray(sd[p + "mlp.fc.weight"].T),
            "proj": jnp.asarray(sd[p + "mlp.proj.weight"].T),
        })
    return params
