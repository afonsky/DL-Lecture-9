# Distance-based Kernels

### Nadaraya--Watson estimators rely on some similarity kernel $\alpha(\mathbf{q}, \mathbf{k})$ relating queries $\mathbf{q}$ to keys $\mathbf{k}$. Some common kernels are:
$$\begin{aligned}
\alpha(\mathbf{q}, \mathbf{k}) & = \exp\left(-\frac{1}{2} \|\mathbf{q} - \mathbf{k}\|^2 \right) && \textrm{Gaussian;} \\
\alpha(\mathbf{q}, \mathbf{k}) & = 1 \textrm{ if } \|\mathbf{q} - \mathbf{k}\| \leq 1 && \textrm{Boxcar;} \\
\alpha(\mathbf{q}, \mathbf{k}) & = \mathop{\mathrm{max}}\left(0, 1 - \|\mathbf{q} - \mathbf{k}\|\right) && \textrm{Epanechikov.}
\end{aligned}$$

<div>
<center>
  <figure>
    <img src="/output_attention-pooling_d5e6b2_18_0.svg" style="width: 480px !important;">
  </figure>
</center>   
</div>

---

# Attention Pooling by Similarity
### All of them lead to the following equation for regression and classification alike:
$$f(\mathbf{q}) = \sum_i \mathbf{v}_i \frac{\alpha(\mathbf{q}, \mathbf{k}_i)}{\sum_j \alpha(\mathbf{q}, \mathbf{k}_j)}$$

* In **Regression** with observations $(\mathbf{x}_i, y_i)$ for features and labels respectively, $\mathbf{v}_i = y_i$ are scalars, $\mathbf{k}_i = \mathbf{x}_i$ are vectors
  * The query $\mathbf{q}$ denotes the new location where $f$ should be evaluated
* In (multiclass) **classification**, we use one-hot-encoding of $y_i$ to obtain $\mathbf{v}_i$
  * Requires no training. Even more so, if we suitably narrow the kernel with increasing amounts of data, the approach is consistent