# Seq2seq with Attention

* Add an additional attention layer to use encoder’s outputs as memory
* The attention output is used as the decoder’s input
<br>
<br>
<div>
<center>
  <figure>
    <img src="/attention_3.png" style="width: 520px !important;">
  </figure>
</center>   
</div>

---

# Encoder/Decoder Details

* The output of the last recurrent layer in the encoder is used
* The attention output is then concatenated with the embedding output to feed into the first recurrent layer in the decoder

<div>
<center>
  <figure>
    <img src="/seq2seq-details-attention.svg" style="width: 520px !important;">
      <figcaption style="color:#b3b3b3ff; font-size: 11px; position: absolute"><br>Image source:
      <a href="https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html">d2l.ai fig. 11.4.2</a>
    </figcaption>
  </figure>
</center>   
</div>