# Multi-Head Attention

### Instead of performing a single attention pooling, $\mathbf{q}$, $\mathbf{k}$ and $\mathbf{v}$ can be transformed with independently learned linear projections. Then these projected $\mathbf{q}$, $\mathbf{k}$ and $\mathbf{v}$ are fed into attention pooling in parallel. In the end, attention-pooling outputs are concatenated and transformed with another learned linear projection to produce the final output.

<div>
<center>
  <figure>
    <img src="/multi-head-attention.svg" style="width: 480px !important;">
      <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute; right:150px"><br>Image source:
      <a href="https://d2l.ai/chapter_attention-mechanisms-and-transformers/multihead-attention.html">d2l.ai fig. 11.5.1</a>
    </figcaption>
  </figure>
</center>   
</div>



---

# Multi-Head Attention Model

### Given a query $\mathbf{q} \in \mathbb{R}^{d_q}$, a key $\mathbf{k} \in \mathbb{R}^{d_k}$, and a value $\mathbf{v} \in \mathbb{R}^{d_v}$, each attention head $\mathbf{h}_i$  ($i = 1, \ldots, h$) is computed as<br> $\mathbf{h}_i = f(\mathbf W_i^{(q)}\mathbf q, \mathbf W_i^{(k)}\mathbf k,\mathbf W_i^{(v)}\mathbf v) \in \mathbb R^{p_v},$

where 
$\mathbf W_i^{(q)}\in\mathbb R^{p_q\times d_q}$,
$\mathbf W_i^{(k)}\in\mathbb R^{p_k\times d_k}$,
and $\mathbf W_i^{(v)}\in\mathbb R^{p_v\times d_v}$<br>
are learnable parameters and
$f$ is attention pooling.

The multi-head attention output
is another linear transformation via 
learnable parameters
$\mathbf W_o\in\mathbb R^{p_o\times h p_v}$
of the concatenation of $h$ heads: $\mathbf W_o \begin{bmatrix}\mathbf h_1\\\vdots\\\mathbf h_h\end{bmatrix} \in \mathbb{R}^{p_o}$

Based on this design, each head may attend
to different parts of the input.