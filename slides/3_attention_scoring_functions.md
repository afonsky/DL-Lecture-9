# Attention Scoring Functions

### Computing the output of attention pooling as a weighted average of values, where weights are computed with the attention scoring function and the softmax operation:
<div>
<center>
  <figure>
    <img src="/qkv.svg" style="width: 520px !important;">
      <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute"><br>Image source:
      <a href="https://d2l.ai/chapter_attention-mechanisms-and-transformers/attention-scoring-functions.html">d2l.ai fig. 11.3.1</a>
    </figcaption>
  </figure>
</center>   
</div>

---

# Dot Product Attention

* Assume the query has the same length as values $\mathbf{q}, \mathbf{k}_i \in \R^{d}$:<br>
$$\alpha(\mathbf{q}, \mathbf{k}_i) = \frac{<\mathbf{q}, \mathbf{k}>}{\sqrt{d}}$$

* Vectorized version
  * $m$ queries $\mathbf{Q} \in \R^{m \times d}$ and $n$ keys $\mathbf{K} \in \R^{n \times d}$:<br>
$$\alpha(\mathbf{Q}, \mathbf{K}) = \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d}}$$

---

# Batch Matrix Multiplication

### Another commonly used operation is to multiply batches of matrices by one another. This comes in handy when we have minibatches of queries, keys, and values. More specifically, assume that 

$$\mathbf{Q} = [\mathbf{Q}_1, \mathbf{Q}_2, \ldots, \mathbf{Q}_n]  \in \mathbb{R}^{n \times a \times b}, \\
    \mathbf{K} = [\mathbf{K}_1, \mathbf{K}_2, \ldots, \mathbf{K}_n]  \in \mathbb{R}^{n \times b \times c}$$

### Then the batch matrix multiplication (BMM) computes the elementwise product

$$\textrm{BMM}(\mathbf{Q}, \mathbf{K}) = [\mathbf{Q}_1 \mathbf{K}_1, \mathbf{Q}_2 \mathbf{K}_2, \ldots, \mathbf{Q}_n \mathbf{K}_n] \in \mathbb{R}^{n \times a \times c}$$

---

# Scaled Dot Product Attention

### In general, dot product attention requires that both the query and the key
have the same vector length, say $d$, even though this can be addressed easily by replacing 
$\mathbf{q}^\top \mathbf{k}$ with $\mathbf{q}^\top \mathbf{M} \mathbf{k}$ where $\mathbf{M}$ is a matrix suitably chosen for translating between both spaces. For now assume that the dimensions match. 
<br>
<br>

In practice, we often think of minibatches for efficiency,
such as computing attention for $n$ queries and $m$ key-value pairs,
where queries and keys are of length $d$
and values are of length $v$. The scaled dot product attention 
of queries $\mathbf Q\in\mathbb R^{n\times d}$,
keys $\mathbf K\in\mathbb R^{m\times d}$,
and values $\mathbf V\in\mathbb R^{m\times v}$
thus can be written as 

$$ \mathrm{softmax}\left(\frac{\mathbf Q \mathbf K^\top }{\sqrt{d}}\right) \mathbf V \in \mathbb{R}^{n\times v}$$

---

# Multilayer Perception Attention

* Learnable parameters $\mathbf{W}_k \in \R^{h \times d_k}$, $\mathbf{W}_q \in \R^{h \times d_q}$, and $\mathbf{v} \in \R^h$:<br>
$$\alpha(\mathbf{k}, \mathbf{q}) = \mathbf{v}^T \tanh(\mathbf{W}_k \mathbf{k} + \mathbf{W}_q \mathbf{q})$$

* Equals to concatenate the key and query, and then feed
into a single hidden-layer perception with hidden size $ℎ$
and output size $1$