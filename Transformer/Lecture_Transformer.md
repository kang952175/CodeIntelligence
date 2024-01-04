# Lecture : Transformer
## [1].Attention

### [1-1].Query, Key, Value

- query : 현재 context를 담고 있는 vector
- key : attention하려는 대상들이 query와 얼마나 연관성이 있는지 계산하기 위한 representation vector
- value : weigthed sum을 할 때 사용되는 references들에 대한 vector

### [1-2].Transformer Architecture Overview
![Transformer1](https://github.com/kang952175/CodeIntelligence/blob/main/Img/Transformer1.png?raw=true)

- 문맥 상에 있는 서로 영향을 받아 자신을 transform함
- 전체적인 문맥에 맞게 자기 자신을 표현하는 방식을 바꾼 contextualized vector가 됨

## [2].More Details on Transformers

### [2-1]. Remaining Questions

1. The outputs are still a sequence of input tokens (with modified representation). What can we do with these?
    1. How can we solve a useful task like classification or regression?
2. How do we train the Transformer model?
    1. Where do we compare the model’s prediction with the ground truth?
    2. Where does the loss arise?
3. Unlike RNN, the order of tokens seems ignored. How can we model it?

### [2-2]. Token Aggregation (Average Pooling)

- The simplest aggregation we may think of is taking average!
    - Taking the mean over all revised token embeddings {$z_1^{(L)},...,z_N^{(L)}$}, we got a single embeddings z that represents the entire input.
- On top of this, we may train a classifier or regressor to solve the target task we are interested in.

![Transformer2](https://github.com/kang952175/CodeIntelligence/blob/main/Img/Transformer2.png?raw=true)

- Averaging may work well if the sequence is not very long and relatively homogeneous.
    - If not, the mean embedding z may not reflect the meaning of the entire input.
    - E.g., Does the average of word embeddings would reflect semantics of the entire sentence?

### [2-2]. Token Aggregation (Classification Token)

- Alternatively, we may rely on the **attention mechanism**
    - The aggregated embedding is trained to selectively attend informative parts within the sequence!
    - Each $z_i$ , however, represents each particular $x_i$, not the entire sequence, because $z_i$ is computed based on the relevance to token i.
- A dummy token (called **classification token**; [CLS]) is appended to the input sequence, and use it as the aggregated embedding.
    - As the dummy input does not convey any meaning, it is not biased to any token.
- 전체를 담는 토큰을 생각함
- [CLS]는 특정 토큰에 대한 특징을 담지 않는 아무 정보 없는 무작위 토큰입니다.

![Transformer3](https://github.com/kang952175/CodeIntelligence/blob/main/Img/Transformer3.png?raw=true)

### [2-3]. Training Transformers

- on the last layer’s output embedding, we put a classifier or regressor.
- The model’s output is compared with the ground truth (GT), and this loss is backpropagated all the way back to the first level.
- Classifier or regressor may be put on the **sequence** or **token level**, depending on the availability of GT labels.

![Transformer4](https://github.com/kang952175/CodeIntelligence/blob/main/Img/Transformer4.png?raw=true)

### [2-4]. Inside the Transformer

- **9개의 스텝으로 진행**이 됩니다.
- Transformer의 input은 sequence data입니다.

**Step 1 : Input Embedding**

- Input is **a sequence of tokens** (e.g., words)
    - Each token is a same-sized vector, represented in a modality-specific way.
    - Examples
        - Text : pre-trained word embeddings
        - Image : fixed size small image patches
        - Video : frame embeddings

### [2-5]. Transformer(Encoder)

**Step 2 : Contextualizing the Word Embedding**

- ★ **input으로 들어오는 모든 토큰들이 자기 자신의 query, key, value embedding을 따로 갖게 됩니다.**
- W는 각 dimension 값을 linear mapping 해주는 matrix를 학습합니다.
- Query, Key, Value representations:
    - For each word, we learn to map it to Q, K, V: instead of using the original embedding, (usually smaller) representation to work like a query, key, and value by linear transformation.

![Transformer5](https://github.com/kang952175/CodeIntelligence/blob/main/Img/Transformer5.png?raw=true)

- At te beginning, Q, K, V are just a random projection of input X.
- As those words are encountered during training, $W^{(Q,K,V)}$will be gradually map X to each so that Q, K ,V to serve as its own purpose.

- **Self-attention**:
    - input sequence로 들어온 자기 자신에서 query, key, value가 다 나옵니다.
    - We will play in this smaller Q, K, V space to attend.
    - For each input word as query (Q), we **compute similarity** with all words as key (K), **including the queried word**, in the input sequence.
    - Then, all words as value (V) are weighted-summed.
        
        → This is the attention value (Z), the **contextualized** new word embedding of **same size**.
        
    

The self-attention calculation in matrix form

![Transformer6](https://github.com/kang952175/CodeIntelligence/blob/main/Img/Transformer6.png?raw=true)

- As the query Q itself is included in the weighted sum, Z tends to **be still self-dominated.**

모두가 자기 자신이 한 번 씩 query가 됩니다.

가령 첫 단어가 query이면 key, value는 자기 자신을 포함한 나머지 모든 단어가 됩니다.

attention은 query를 기준으로 value들의 weighted sum을 구하는 것입니다.

weight은 key와 query의 similarity로 정해집니다.

★즉, query를 기준으로 references의 similarity를 계산하여 그 references의 value 로 사용되는 vector들의 weighted sum을 구합니다.

★Z는 query를 기준으로 각각의 단어들의 embedding 정보를 합쳐 contextualized하는 과정입니다.

- **Multi-head** Self-attention:
    - Having multiple projections to Q, K, V is beneficial
    - Allows the model to jointly attend to information from different representation subspaces at different positions,
        
        ![Transformer7](https://github.com/kang952175/CodeIntelligence/blob/main/Img/Transformer7.png?raw=true)

        - 문맥에 따라서 it 같은 중의적인 표현이 존재하는 경우를 구현하기 위함
    - Multiple self-attentions output multiple attention values ($Z_0, Z_1,…,Z_{k-1}$)
    - Simply concatenate them, then linearly transform it back to the original input size with $W^o$.
    - 최대한 8개의 head로 단어나 문맥에서 중의적인 부분을 처리하는데 충분했다. (자연어)
    
    ![Transformer8](https://github.com/kang952175/CodeIntelligence/blob/main/Img/Transformer8.png?raw=true)

    
    Z는 input과 같은 길이와 크기의 sequence가 나옵니다 
    
- one-step of Encoder
- w가 서로 다르게 initialize되어서 서로 다른 정보를 학습하도록 함

![Transformer9](https://github.com/kang952175/CodeIntelligence/blob/main/Img/Transformer9.png?raw=true)


- R is Z (new input)

**step 3 : Feed-forward layer**

- The new, contextualized word embedding goes through an additional FC layer.
    
    $$
    FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2
    $$
    
    - Applied separately and identically : there is no cross-token dependency
    - 각자 자기 토큰 안에서 자신을 표현할 수 있는 방식을 배우는 capacity를 주자
    - 최종적으로 가장 잘 표현할 수 있도록 자기 자신만 한 번 transform을 수행

- The output is still a **same-size contextualized** token embedding
- Residual connection and layer normalization is added at the end of multi-head self-attention and FC layer.
    - Residual connection : 복잡도가 충분하여 필요하지 않으면 업데이트 되지 않도록 네트워크 구성
    - layer normalization : 학습에 도움이 되도록 해줌

> Stacked self-attention blocks
> 
- 전체 문맥에 더 contextulaized 되도록 여러 층을 통과시킴
- Multiple (N) self-attention blocks are stacked
    - The lowest block takes a pre-trained embedding as input.
    - Blocks stacked after the first one take the output embedding of the previous one.
- The last layer outputs a **same**-**length sequence** of trnsformed tokens as input, where each of them is also a **same**-**sized** contextualized embedding.
- In the end, the output is a one-to-one match, only the representation of the vector has been changed to contextualize the other words. ⇒ Transformer! (형식은 그대로 벡터 값이 변형됨)

> Positional encoding
> 
- Unlike RNNs, the tokens had no concept of order.
    - Each token just attended other tokens in the sequence
- To inject the order information, Transformer adds positional encoding in the input:
    - Same words at different locations will have different overall representations
    - With sinusoidal encoding, we can deal with arbitrarily long sequences at test time.

$$
PE_{(pos, 2i)} = sin(pos/10000^{2i/d_{model}}) \newline PE_{(pos, 2i+1)} = cos(pos/10000^{2i/d_{model}})
$$

문장의 특징 상 비슷한 위치에서는 비슷한 성질 주어, 동사 등의 의미가 있다.

점진적으로 변화할 수 있도록 posion vector를 만들었다.

단어들의 순서를 표현하기 위한 방법을 위해 사용하였다.

![Transformer10](https://github.com/kang952175/CodeIntelligence/blob/main/Img/Transformer10.png?raw=true)

- 문장의 표현 상 필요에 의해서 셋팅함
- 임베딩 크기 절반 -1 까지 올라감
- With $i = 0, ..., d_{model}/2-1$ (0부터 31):
    - $2i/d_{model}$ (64) : gradually **increase** from 0 to 1
    - $1/10000^{2i/d}$ : gradually **decrease** from 1 to 0
    - With higher i (rear indices), the input ($pos/10000^{2i/d}$)changes slowly. 
    With lower i (front indices), the input ($pos/10000^{2i/d}$) changes more frequently.
    - **No two different indices have same encoding.**
        - 모든 인코딩은 서로 달라야 합니다.
        - 앞은 자주 뒤는 천천히 바뀌게 하여 모든 값이 항상 다르게 만들어 특정 위치를 unique하게 표현
- **Adjacent pos** (the order of word in a sentence) has **similar positional encoding.**
    - There is no absolute binding between the order of a word and its role in the sentence.
    (With longer subject, the verb is pushed to a rearer index.)
    - 상대적인 순서가 중요하며 동사 입장에서 몇 번째 단어가 될지는 주어의 길이에 따라 달라지며 고정되어 있지 않다.
    - 인접한 포지션 인코딩은 비슷한 값을 지니게 만듬

positional encoding은 sequence 들어가는 token들의 순서를 모델링하기 위함

- 조건1) 어떤 index도 같은 인코딩을 갖지 않아야 한다.
- 조건2) 인접한 인코딩끼리는 비슷한 값을 가져야 한다.

### [2-6].Transformer (Decoder)

- ★ sequence of token 이 들어가고 그 길이가 계속 같은 output으로 나옵니다.

![Transformer11](https://github.com/kang952175/CodeIntelligence/blob/main/Img/Transformer11.png?raw=true)

**Step 4 : Decoder input**

- Given $Z = \{z_1, ...,z_n\}$(the encoder output), generates an output sequence **auto**-**regressively**.
    - Input으로 target language의 sequence가 들어갑니다.
    - Consumes the previously generated symbol as additional input when generating the next.
- Positional encoding is applied in the same manner as the encoder.

**Step 5 : Masked Multi-head Self-attention**

- The input sequence (here, the generated output so far) is fed into multi-head self-attention layer as in the encoder
- Since we have no idea **after the current time step**, they are masked out.
- Other than this masking, this is exactly the same as multi-head self-attention layer in the encoder:
    - Each token in the input is contextualized and transformed

**Step 6 : Encoder-Decoder attention**

- Now, the input attends the encoder output:
    - Q: the query from decoder
    - K, V: the key and value from encoder
    - Other than this, **Same as multi-head self-attention** layer.
    - **No masking** in this layer, as it is okeay (and necessary) to look at the entire encoded sequence.
    - 원문의 reference를 참고합니다. (No masking!)

![Transformer12](https://github.com/kang952175/CodeIntelligence/blob/main/Img/Transformer12.png?raw=true)


**Step 7 : Feed-forward layer**

- same as encdoer
- Residual, layer normalization: same as encoder
- N stacked blocks: same as encoder
    - The last layer output is fed as input in the next time step.

decoder 1) 자기 문장 구조 파악 2) 원문을 보기 위함 3) 자신을 잘 표현하는 capacity 학습

**Step 8 : Linear layer**

- Maps the output embedding to **class** scores.
    - Output size : equal to the vocabulary size

**Step 9 : Softmax layer**

- Takes **softmax** to the class scores to make them as a probability
- These scores (that are summed to 1 ) are compared with 1-hot-encoded ground truth.
    - Then, backprops with some loss (e.g. cross-entropy)
- These decoding steps are repeated until the next word is predicted as [EOS] (End of Sentence)
- The output sentence may be chosen greedily (always with the top one), or deferred with top-k choices (called **beam search**)

## [3]. Transformer for Image Data

### [3-1]. ViT : Vision Transformer

- 이미지의 sequential한 정보를 어떻게 만들까?
- 이미지 구성도 유기적으로 작은 이미지의 연결된 정보이다.
- The standard Transformer model is directly applied to images:
    - An image is split into 16 x 16 patches. (Each token is **a 16 x 16 image patch** instead of a word.)
    - The sequence of linear embeddings of these patches are fed into a **Transformer**.
    - Image patches are treated on the same way as tokens (words)
    - Eventually, an MLP is added on top of the [CLS] **token** to classify the input image.

![ViT1](https://github.com/kang952175/CodeIntelligence/blob/main/Img/ViT1.png?raw=true)


transformer의 input : sequence data (input을 split 하여 그 sub components의 sequnce로 들어가 서로를  표현하는 방식을 배움 )

이미지는 딱 한 장 ⇒ How to split into a sequence?

224 x 224 ⇒ 16 x 16 patches

[CLS] : 더미 토큰으로 특정 하나의 bias가 되지 않는 이미지 전체를 표현하는 토큰

L : layer의 개수

![ViT2](https://github.com/kang952175/CodeIntelligence/blob/main/Img/ViT2.png?raw=true)


Input! : sequence of token embedding (p by p patches)

pixel 차원이 크기 때문에 linear projection으로 Mapping시켜줍니다. ⇒  token

pixel level의 표현을 embedding으로 하는 일반적으로 학습하게 합니다.

positional Encoding은 learnable encoding

- ViT outperforms previous SOTA (ResNet152)
- But, It takes 300 days with 8 TPUv3 cores (연산을 위한 하드웨어 in google)
    - 2500일
    - $8 x 24 hr/day x 2500 day = $480.000 to train this model once! (5억)
- 인간에게 자명한 task를 돈을 엄청 태우는게 실상

- ViT performs well only when trained on an extremely large dataset (e.g., JFT-300M). why?
    - ViT does NOT imply any inductive bias (spatial locality & positional invariance) of CNNs.
    - It needs to learn those **purely from the data.** 
    → It requires large amount of examples.
    - Once sufficient training examples provided, however, it can outperform CNN-based models, as it is **capable of modeling hard cases beyond spatial locality.**

- inductive bias 가 없다.
- CNN이 갖는 spatial locality, positional invariance로 비주얼한 패턴을 확인
- 장기적으로 attention을 학습하는 경우 국소적인 학습도 깨우치더라

### [4-2]. ViT : Position Embeddings

![ViT3](https://github.com/kang952175/CodeIntelligence/blob/main/Img/ViT3.png?raw=true)

- ViT learns to encode distance within the image in the similarity of position embeddings.
    - Closer patches tend to have more similar position embeddings
- The row-column structure appears.
    - Patches in the same row/column have similar embeddings, automatically learned from data.
- Hand-crafted 2D-aware embedding variants do not yield improvements for this reason.
