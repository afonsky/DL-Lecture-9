# Self-Attention

### Every token is attending to each other token

Given a sequence of input tokens
$\mathbf{x}_1, \ldots, \mathbf{x}_n$ where any $\mathbf{x}_i \in \mathbb{R}^d$ ($1 \leq i \leq n$),
its self-attention outputs
a sequence of the same length
$\mathbf{y}_1, \ldots, \mathbf{y}_n$,
where

$$\mathbf{y}_i = f(\mathbf{x}_i, (\mathbf{x}_1, \mathbf{x}_1), \ldots, (\mathbf{x}_n, \mathbf{x}_n)) \in \mathbb{R}^d$$

---

# Comparing CNNs, RNNs, and Self-Attention

<div>
<center>
  <figure>
    <img src="/cnn-rnn-self-attention.svg" style="width: 400px !important;">
      <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute; right:150px"><br>Image source:
      <a href="https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html">d2l.ai fig. 11.6.1</a>
    </figcaption>
  </figure>
</center>   
</div>
<br>

* Computational complexity of the convolutional layer / recurrent layer / self-attention is  $\mathcal{O}(knd^2)$ / $\mathcal{O}(nd^2)$ / $\mathcal{O}(n^2d)$
* Self-attention computation can be parallel with $\mathcal{O}(1)$ sequential operations

---

# Positional Encoding

### The dominant approach for preserving information about the order of tokens is to represent this to the model as an additional input associated with each token. These inputs are called **positional encodings**, and they can either be learned or fixed *a priori*.

Suppose that the input representation 
$\mathbf{X} \in \mathbb{R}^{n \times d}$ 
contains the $d$-dimensional embeddings 
for $n$ tokens of a sequence.
The positional encoding outputs
$\mathbf{X} + \mathbf{P}$
using a positional embedding matrix 
$\mathbf{P} \in \mathbb{R}^{n \times d}$ of the same shape,
whose element on the $i^\textrm{th}$ row 
and the $(2j)^\textrm{th}$
or the $(2j + 1)^\textrm{th}$ column is
<div class="grid grid-cols-[3fr_2fr] gap-3">
<div>

$p_{i, 2j} = \sin(\frac{i}{10000^{2j/d}})$<br>
<br>
$p_{i, 2j+1} = \cos(\frac{i}{10000^{2j/d}})$
</div>
<div>
    <figure>
    <img src="/output_self-attention-and-positional-encoding_ce9eb6_48_0.svg" style="width: 300px !important;">
  </figure>
</div>
</div>

---

# Relative Positional Information

### Besides capturing absolute positional information, the positional encoding also allows a model to learn to attend by **relative positions**. This is because for any fixed position offset $\delta$, the positional encoding at position $i + \delta$ can be represented by a linear projection of that at position $i$.

Denoting
$\omega_j = 1/10000^{2j/d}$,
any pair of $(p_{i, 2j}, p_{i, 2j+1})$ can 
be linearly projected to $(p_{i+\delta, 2j}, p_{i+\delta, 2j+1})$
for any fixed offset $\delta$:
$$\scriptsize\begin{aligned}
\begin{bmatrix} \cos(\delta \omega_j) & \sin(\delta \omega_j) \\  -\sin(\delta \omega_j) & \cos(\delta \omega_j) \\ \end{bmatrix}
\begin{bmatrix} p_{i, 2j} \\  p_{i, 2j+1} \\ \end{bmatrix}
=&\begin{bmatrix} \cos(\delta \omega_j) \sin(i \omega_j) + \sin(\delta \omega_j) \cos(i \omega_j) \\  -\sin(\delta \omega_j) \sin(i \omega_j) + \cos(\delta \omega_j) \cos(i \omega_j) \\ \end{bmatrix}\\
=&\begin{bmatrix} \sin\left((i+\delta) \omega_j\right) \\  \cos\left((i+\delta) \omega_j\right) \\ \end{bmatrix} = \begin{bmatrix} p_{i+\delta, 2j} \\  p_{i+\delta, 2j+1} \\ \end{bmatrix},
\end{aligned}$$
#### where the $2\times 2$ projection matrix does not depend on any position index $i$.