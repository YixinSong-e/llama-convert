#!/usr/bin/env python
# coding=utf-8
from transformers import LlamaConfig
import torch, os
import json
from collections import OrderedDict

ver_layernum = [
    "7b",
    "13b",
    "30b",
    "65b",
]

ver = ver_layernum[0]

# inpath = f"../results/llama-{ver}-hf"
outpath = f"./7B"

# hf_config = LlamaConfig.from_pretrained(inpath)
config = {
    'dim_model': 4096,
    'dim_ff': 11008,
    'num_layers': 32,
    'num_heads':32,
    'dim_head': 128,
}
# with open(os.path.join(outpath, "config.json"), 'w') as f:
#     json.dump(config, f)

layernum = config['num_layers']

model_hf = OrderedDict()
# for i in range(1, layernum + 2):
#     part = torch.load(os.path.join(inpath, f"pytorch_model-{i:05d}-of-000{layernum+1}.bin"))
#     model_hf.update(part)
model_hf = torch.load("2000.pkl")['model']
out = OrderedDict()

# out["input_embedding.weight"] = model_hf['model.embed_tokens.weight'].contiguous()
# out["encoder.output_layernorm.weight"] = model_hf['model.norm.weight'].contiguous()
# out['output_projection.weight'] = model_hf['lm_head.weight'].contiguous()

out["tok_embeddings.weight"] = model_hf['model.input_embedding.weight'].contiguous()
out["norm.weight"] = model_hf['model.encoder.output_layernorm.weight'].contiguous()
out['output.weight'] = model_hf['model.output_projection.weight'].contiguous()
for lnum in range(layernum):
    # hf_pfx = f"model.layers.{lnum}"
    # bmt_pfx = f"encoder.layers.{lnum}"
    
    bmt_pfx = f"layers.{lnum}"
    hf_pfx = f"model.encoder.layers.{lnum}"
    # out[f"{bmt_pfx}.self_att.layernorm_before_attention.weight"] = model_hf[f"{hf_pfx}.input_layernorm.weight"].contiguous()
    out[f"{bmt_pfx}.attention_norm.weight"] = model_hf[f"{hf_pfx}.self_att.layernorm_before_attention.weight"].contiguous()

    # out[f"{bmt_pfx}.self_att.self_attention.project_q.weight"] = model_hf[f"{hf_pfx}.self_attn.q_proj.weight"].contiguous()
    out[f"{bmt_pfx}.attention.wq.weight"] = model_hf[f"{hf_pfx}.self_att.self_attention.project_q.weight"].contiguous()
    # out[f"{bmt_pfx}.self_att.self_attention.project_k.weight"] = model_hf[f"{hf_pfx}.self_attn.k_proj.weight"].contiguous()
    out[f"{bmt_pfx}.attention.wk.weight"] = model_hf[f"{hf_pfx}.self_att.self_attention.project_k.weight"].contiguous()
    # out[f"{bmt_pfx}.self_att.self_attention.project_v.weight"] = model_hf[f"{hf_pfx}.self_attn.v_proj.weight"].contiguous()
    out[f"{bmt_pfx}.attention.wv.weight"] = model_hf[f"{hf_pfx}.self_att.self_attention.project_v.weight"].contiguous()
    # out[f"{bmt_pfx}.self_att.self_attention.attention_out.weight"] = model_hf[f"{hf_pfx}.self_attn.o_proj.weight"].contiguous()
    out[f"{bmt_pfx}.attention.wo.weight"] = model_hf[f"{hf_pfx}.self_att.self_attention.attention_out.weight"].contiguous()

    # out[f"{bmt_pfx}.ffn.layernorm_before_ffn.weight"] = model_hf[f"{hf_pfx}.post_attention_layernorm.weight"].contiguous()
    out[f"{bmt_pfx}.ffn_norm.weight"] = model_hf[f"{hf_pfx}.ffn.layernorm_before_ffn.weight"].contiguous()

    # out[f"{bmt_pfx}.ffn.ffn.w_in.w_0.weight"] = model_hf[f"{hf_pfx}.mlp.gate_proj.weight"].contiguous()
    out[f"{bmt_pfx}.feed_forward.w1.weight"] = model_hf[f"{hf_pfx}.ffn.ffn.w_in.w_0.weight"].contiguous()
    # out[f"{bmt_pfx}.ffn.ffn.w_in.w_1.weight"] = model_hf[f"{hf_pfx}.mlp.up_proj.weight"].contiguous()
    out[f"{bmt_pfx}.feed_forward.w3.weight"] = model_hf[f"{hf_pfx}.ffn.ffn.w_in.w_1.weight"].contiguous()

    # out[f"{bmt_pfx}.ffn.ffn.w_out.weight"] = model_hf[f"{hf_pfx}.mlp.down_proj.weight"].contiguous()
    out[f"{bmt_pfx}.feed_forward.w2.weight"] = model_hf[f"{hf_pfx}.ffn.ffn.w_out.weight"].contiguous()
    out[f"{bmt_pfx}.attention.inner_attention.rope.freqs"] = model_hf[f"model.position_bias.inv_freq"].contiguous()
    
    
for key in out:
    out[key] = out[key].half()

if not os.path.exists(outpath):
    os.makedirs(outpath)
torch.save(out, os.path.join(outpath, "consolidated.00.pth"))
