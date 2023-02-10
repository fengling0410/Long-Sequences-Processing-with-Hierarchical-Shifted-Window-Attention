# Long-Sequences-Processing-with-Hierarchical-Shifted-Window-Attention

### Author: Kexin Yang, Ling Feng

## Abstract
Long sequences processing has been a challenging task for traditional transformers as the full self-attention mechanism scales quadratically with respect to the input sequence length. Different designs over attention have been applied to address this issue to reduce computational complexity and take into account both global and local features of the document. Here, we describe an approach to use shifted windows in attention calculation to construct a hierarchical structure in processing long sequences. Our work is inspired by the unprecedented performance of Swin Transformer in Computer Vision tasks. Hierarchical structure reduces computational complexity and improves training and inference speed, while shifted window attention enables communication across windows. Our Hierarchical Shifted Window (HSW) method has achieved the highest classification F1 score in almost all the categories in Arxiv Dataset among baseline models and has reduced the training and inference speed by 10 times compared with Longformer. We hope our work could shed some light on unified modeling in both CV and NLP, as Swin Transformer shows great performances in both fields.

## Models
### Baseline Model - Longformer
Longformer (Beltagy et al., 2020) aims to reduce the computation complexity of BERT-like transformers to handle long sequences. Given a sequence length $n$, instead of using full $n^2$ attention, it substitutes it with sliding window attention and several global attention tokens. The original repository of longerform can be accessed from this [link](https://github.com/allenai/longformer).

### Our Model - Hierarchical Shifted Window Attention
Our approach has two folds. First, we proposed a \textbf{hierarchical structure} for faster training and inference process and reduced computational requirements. In the first layer, each of the 4096 words is treated as token of size 96. Patch merging is used to down-sample the feature size from layer 1 to layer 2. After three times of down-sampling, the final size of the long sequence representation has become 64 x 768. In comparison, the original transformer model and the Longformer model have a final representation of size 4096 x 768. Therefore, our hierarchical structure greatly reduces training and inference time by using only 64 tokens to summarize the sequence. Additionally, after each down-sample operation, the tokens in the next layer get to a higher level representation. The lower level of our hierarchical structure extracts word/sentence level information, and the higher level attentions extract section/article level information. The pyramid representation of language information is more useful for less generative tasks, including text classification.

\noindent Second, we proposed the use of \textbf{shifted window attention} in long sequence processing to enable communication across attention windows. The previously proposed window attention methods are effective at reducing computational complexity, but they limit the transformer's ability to model long-range dependencies. We adapted the idea of shifted window attention from the Swin Transformer model for Computer Vision, where the attention is shifted to the right by half of the window length. Through this shifted attention mechanism, tokens within a certain attention window can perform attention with tokens in the next window. In our implementation, for layer 1, the first 64 tokens are concatenated with the last 64 tokens to perform self-attention, maintaining the computational intensity before and after window shifting. In order to prevent the first tokens from performing attention with the last tokens, an attention mask is added to the self-attention matrix, as stated in the [Swin Transformer paper](https://arxiv.org/abs/2103.14030). 

![Alt text](images/attentionwindow.png?raw=true "Attention Window Design")

![Alt text](images/hierarchical.png?raw=true "Hierarchical Structure")

## Dataset
We used the Arxiv Classification Dataset (He et al., 2019) in this project. This Dataset is created specifically for the purpose of long-text classification with 11 scientific categories and each category contains around 2500 papers.

## Experiment Results
Our HSW model shows better results than Longformer, the previous SOTA model on long-sequence text, when training on the Arvix Dataset on the multi-class classification task. While the results we obtained probably do not reflect these models' full potential due to our limited computation resources and relatively small training epoch, they provide strong evidence that the shifted window attention mechanism, originally designed for CV tasks, can also be a good solution to solve the high computation complexity issue in long sequence text. 


![Alt text](images/longformer.png?raw=true "Longformer classification accuracy")

![Alt text](images/HSAmodel.png?raw=true "HSA model classification accuracy")

![Alt text](images/runtime.png?raw=true "Runtime comparison")