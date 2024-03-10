# Motivation

### Each generated token might be related to different source tokens
<br>
<br>

<div>
<center>
  <figure>
    <img src="/attention_1.png" style="width: 920px !important;">
  </figure>
</center>   
</div>

---

# Attention Layer
<br>
<br>

### An attention layer explicitly select related information
<br>

<div class="grid grid-cols-[2fr_2fr] gap-3">
<div>
<br>

* Its memory consists of key-value pairs

* The output will be close to the values whose keys are similar to the query 
</div>
<div>
<center>
  <figure>
    <img src="/attention_2.png" style="width: 400px !important;">
  </figure>
</center>    
</div>
</div>

---

# Attention Layer

### Denote by $\mathbf{q}$ a *query*.
* Assume query $\mathbf{q} \in \R^{d_q}$ and the memory $\mathcal{D} \stackrel{\textrm{def}}{=} \{(\mathbf{k}_1, \mathbf{v}_1), \ldots (\mathbf{k}_m, \mathbf{v}_m)\}$
  * with $\mathbf{k}_i \in \R^{d_k}$ and $\mathbf{v}_i \in \R^{d_v}$

* Compute $m$ scores $a_1, ..., a_m$ with $a_i = \alpha(\mathbf{q}, \mathbf{k}_i)$ - **attention weights**
  * Vary $\alpha$ to get different attention layers

* Use softmax to obtain the attention weights:<br>
$b_1, ..., b_m = \mathrm{softmax}(a_1, ..., a_m)$

* The output is a weighted sum of values:
$~\mathbf{o} = \sum\limits_{i=1}^m b_i \mathbf{v}_i$

---

# Attention Pooling

### We can define the **attention** over $\mathcal{D}$ as

$$\textrm{Attention}(\mathbf{q}, \mathcal{D}) \stackrel{\textrm{def}}{=} \sum_{i=1}^m \alpha(\mathbf{q}, \mathbf{k}_i) \mathbf{v}_i,$$

The name **attention** derives from the fact that the operation pays particular attention to the terms for which the weight $\alpha$ is significant (i.e., large). As such, the attention over $\mathcal{D}$ generates a linear combination of values contained in the database. In fact, this contains the above example as a special case where all but one weight is zero.

---

# Special Cases of Attention Weights

* $\alpha(\mathbf{q}, \mathbf{k}_i)$ are nonnegative
  * The output of the attention mechanism is contained in the convex cone spanned by the values $\mathbf{v}_i$
* $\alpha(\mathbf{q}, \mathbf{k}_i)$ form a convex combination
  * i.e., $\sum_i \alpha(\mathbf{q}, \mathbf{k}_i) = 1$ and $\alpha(\mathbf{q}, \mathbf{k}_i) \geq 0$ for all $i$
  * This is the **most common setting** in deep learning
* Exactly one of the weights $\alpha(\mathbf{q}, \mathbf{k}_i)$ is $1$, while all others are $0$
  * Traditional database query
* All weights are equal, i.e., $\alpha(\mathbf{q}, \mathbf{k}_i) = \frac{1}{m}$ for all $i$
  * Averaging across the entire database, also called **average pooling** in DL

---

# Attention Weights Form a Convex Combination

### A common strategy for ensuring that the weights sum up to $1$ is to normalize them via 

$$\alpha(\mathbf{q}, \mathbf{k}_i) = \frac{\alpha(\mathbf{q}, \mathbf{k}_i)}{\sum\limits_j \alpha(\mathbf{q}, \mathbf{k}_j)}$$
In particular, to ensure that the weights are also nonnegative, one can resort to exponentiation. This means that we can now pick **any** function  $a(\mathbf{q}, \mathbf{k})$ and then apply the softmax operation used for multinomial models to it via

$$\alpha(\mathbf{q}, \mathbf{k}_i) = \frac{\exp(a(\mathbf{q}, \mathbf{k}_i))}{\sum\limits_j \exp(a(\mathbf{q}, \mathbf{k}_j))}$$

---

# Attention Layer

<div>
<center>
  <figure>
    <img src="/qkv.svg" style="width: 520px !important;">
      <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute"><br>Image source:
      <a href="https://d2l.ai/chapter_attention-mechanisms-and-transformers/queries-keys-values.html">d2l.ai fig. 11.1.1</a>
    </figcaption>
  </figure>
</center>   
</div>