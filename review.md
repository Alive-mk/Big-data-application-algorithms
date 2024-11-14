![image-20241113190559495](pictures\image-20241113190559495.png)

# 1. 第一章

## 1.1 Key-Value pairs

```
Data element e∈D has key and value (e.key,e.value)
```

<img src="pictures\image-20241110105941435.png" alt="image-20241110105941435" style="zoom: 50%;" />

## 1.2 Summary Structures (Sketches)

<img src="pictures\image-20241110110425847.png" alt="image-20241110110425847" style="zoom:50%;" />

<img src="pictures\image-20241110110605689.png" alt="image-20241110110605689" style="zoom: 67%;" />

### 1.2.1 `Composable sketches`

<img src="pictures\image-20241110110830540.png" alt="image-20241110110830540" style="zoom:50%;" />



### 1.2.2 `Streaming sketches`

<img src="pictures\image-20241110111245791.png" alt="image-20241110111245791" style="zoom:50%;" />

### 1.2.3 `Sketch API`

<img src="pictures\image-20241110111501453.png" alt="image-20241110111501453" style="zoom:50%;" />

<img src="pictures\image-20241110111518949.png" alt="image-20241110111518949" style="zoom:67%;" />

### 1.2.4 example

<img src="pictures\image-20241110111659913.png" alt="image-20241110111659913" style="zoom:50%;" />

max:不支持删除操作，因为删除元素后无法准确地维护最大值。

## 1.3 Frequent Keys-MG

MG算法（Misra-Gries Algorithm）是一种用于 **频繁元素检测** 的流处理算法，特别适用于大规模数据流中识别**出现次数较高的元素**。该算法在有限的空间内运行，能够在不保存所有数据的情况下，近似找出数据流中的频繁元素。

**Find the keys that occur very often.**

<img src="C:\Users\LX\Desktop\大数据应用算法\pictures\image-20241110112130028.png" alt="image-20241110112130028" style="zoom:50%;" />

幻灯片中还提到了 **`Zipf`定律**，即第 i 个频繁键的出现频率 ∝i的s次方分之一，这意味着少数键占据大部分频率。例如，前 10% 的键可能在 90% 的元素中出现。`Zipf`定律在许多场景中适用，例如城市人口分布、社交网络中的用户行为等。

在这张幻灯片中，s是 `Zipf`定律中的指数参数，它控制了频率分布的陡峭程度。

<img src="C:\Users\LX\AppData\Roaming\Typora\typora-user-images\image-20241111144446679.png" alt="image-20241111144446679" style="zoom: 50%;" />

<img src="C:\Users\LX\AppData\Roaming\Typora\typora-user-images\image-20241111145222086.png" alt="image-20241111145222086" style="zoom:50%;" />

我们限制“减少”步骤的次数

为了保证误差的上限，算法限制了“减少”步骤的次数。具体说明如下：

- 每次减少步骤会从数据结构中移除 k 个“计数”，同时还包含一个输入元素，这样总共会产生 k+1 个未计入的元素。

- 因此，每次减少步骤实际上减少了 k+1 个未被计数的元素。由此可以推出减少步骤的总次数应满足 
  $$
  \leq \frac{m - m'}{k + 1}
  $$



**估计准确性**：某个元素的频率估计值最多比真实频率低 
$$
\frac{m - m'}{k+1}
$$
其中：

<img src="C:\Users\LX\AppData\Roaming\Typora\typora-user-images\image-20241111152324313.png" alt="image-20241111152324313" style="zoom:50%;" />

这个误差界限表示估计的频率值最多比真实频率低 
$$
\frac{m - m'}{k+1}
$$
**频率条件**：对于频率远大于 
$$
\frac{m - m'}{k+1}
$$
的元素 x，MG sketch 能够提供良好的频率估计。这意味着它对在数据流中出现频率较高的元素有较好的效果。

**误差界限的比例关系**：误差界限与 k成反比，即随着 k的增加，误差会减小。这反映了 sketch 大小和估计质量之间的权衡。

**Zipf 定律**：MG sketch 有效的原因在于典型的频率分布通常符合 Zipf 定律，即**数据流中只有少数元素会非常流行**，大部分元素的出现频率较低。

**Merging two Misra Gries Sketches**

<img src="C:\Users\LX\AppData\Roaming\Typora\typora-user-images\image-20241111170904319.png" alt="image-20241111170904319" style="zoom: 33%;" />

<img src="C:\Users\LX\AppData\Roaming\Typora\typora-user-images\image-20241111170919328.png" alt="image-20241111170919328" style="zoom: 33%;" />

<img src="C:\Users\LX\AppData\Roaming\Typora\typora-user-images\image-20241111171132845.png" alt="image-20241111171132845" style="zoom: 67%;" />

声明2的证明：

<img src="C:\Users\LX\AppData\Roaming\Typora\typora-user-images\image-20241111172031263.png" alt="image-20241111172031263" style="zoom: 67%;" />

<img src="C:\Users\LX\AppData\Roaming\Typora\typora-user-images\image-20241111171525596.png" alt="image-20241111171525596" style="zoom: 67%;" />

<img src="C:\Users\LX\AppData\Roaming\Typora\typora-user-images\image-20241111172058829.png" alt="image-20241111172058829" style="zoom: 33%;" />

## 1.4 Set membership-Bloom Filters

<img src="C:\Users\LX\AppData\Roaming\Typora\typora-user-images\image-20241114085156889.png" alt="image-20241114085156889" style="zoom:67%;" />

<img src="C:\Users\LX\AppData\Roaming\Typora\typora-user-images\image-20241114085217280.png" alt="image-20241114085217280" style="zoom:67%;" />

**广泛应用**：布隆过滤器在许多应用中都非常流行，尤其适合**快速判断元素是否属于一个集合。**

**概率性数据结构**：布隆过滤器是一种概率性的数据结构，允许一定的错误率。

**减少存储需求**：它将每个键的表示大小减少到少量的位（通常是每个键 8 位），大大节省了存储空间。

**可能出现误报，但不会漏报**：布隆过滤器可能会错误地报告一个不存在的元素为存在（误报），**但不会漏掉实际存在的元素。**

**大小与误报率的权衡**：布隆过滤器的大小和误报率之间存在权衡关系。增大过滤器大小可以降低误报率，但会消耗更多的存储空间。

**可组合性**：布隆过滤器具有可组合性，可以将多个布隆过滤器合并以检查集合的联合。

**依赖独立的随机哈希函数**：布隆过滤器的分析依赖于使用独立的随机哈希函数。在实际应用中，这些哈希函数的效果较好，但理论上仍存在一些问题。

<img src="pictures\image-20241111190058752.png" alt="image-20241111190058752" style="zoom: 33%;" />

<img src="pictures\image-20241111190808448.png" alt="image-20241111190808448" style="zoom: 33%;" />

m：布隆过滤器的总位数（结构大小）。

k：使用的哈希函数数量。

n：插入的不同键的数量。

<img src="C:\Users\LX\AppData\Roaming\Typora\typora-user-images\image-20241111191032202.png" alt="image-20241111191032202" style="zoom: 67%;" />

<img src="C:\Users\LX\AppData\Roaming\Typora\typora-user-images\image-20241111191137069.png" alt="image-20241111191137069" style="zoom:50%;" />

## 1.5 Simple Counting-Morris Counter

Morris Counter 是一种 **概率性计数算法**，相比传统的计数器（需要随着计数增长线性增长存储空间），它在空间效率上非常高效。Morris Counter 的核心思想是 **在每一步概率性地决定是否增加计数器的值**，从而用对数空间表示大数。

<img src="C:\Users\LX\AppData\Roaming\Typora\typora-user-images\image-20241111193634738.png" alt="image-20241111193634738" style="zoom: 33%;" />

**Morris Counter 的工作原理**

1. **基本思路**：

   - Morris Counter 使用 loglogn 位来近似计数 n 而不是 logn 位。
   - 通过概率性增量来实现更节省空间的计数。

2. **初始化**：

   - 将计数器 s 初始化为 0。

3. **增量**：

   - 每当遇到一个新元素时，以概率 
     $$
     2^{-s}
     $$
     来增加 s 的值。这意味着 s **增加的频率随着计数增加而减少**，从而实现对大数的近似。

4. **查询**：

   - 查询当前计数的估计值时，返回
     $$
     2^s - 1
     $$
      这是对实际计数的一个近似值。

**Morris Counter** 的**无偏性（Unbiasedness）**，即其计数估计值如何在**数学期望上接近真实值**。

**无偏性目标**：证明每次增量操作后估计值的期望增量始终为 1。

<img src="pictures\image-20241111202621998.png" alt="image-20241111202621998" style="zoom: 67%;" />

**方差分析：**

<img src="pictures\image-20241111202816262.png" alt="image-20241111202816262" style="zoom: 67%;" />

**通过平均减少方差：**

<img src="pictures\image-20241111202918374.png" alt="image-20241111202918374" style="zoom:67%;" />

**Morris Counter 通过独立计数器减少方差：**

<img src="pictures\image-20241111203104892.png" alt="image-20241111203104892" style="zoom:67%;" />

**通过基数改变减少方差：**

<img src="pictures\image-20241111203154335.png" alt="image-20241111203154335" style="zoom:67%;" />

**Weighted  Morris Counter：**

<img src="pictures\image-20241111203702575.png" alt="image-20241111203702575" style="zoom:67%;" />

Morris Counter 适用于需要节省空间且对计数精度要求不高的场景。例如：

- **网络流量计数**：在大规模流量监控中，只需要近似计数即可，不需要精确的计数。
- **用户行为分析**：在一些简单的统计任务中，例如用户点击量、访问量等场景，可以用近似值估算。
- **物联网（IoT）数据处理**：在资源受限的 IoT 设备中，空间效率尤为重要，Morris Counter 可以提供一种轻量化的计数方式。

# 2. 第二章

## 2.1 MinHash Sketch

<img src="C:\Users\LX\AppData\Roaming\Typora\typora-user-images\image-20241111205010477.png" alt="image-20241111205010477" style="zoom:67%;" />

<img src="C:\Users\LX\AppData\Roaming\Typora\typora-user-images\image-20241111205105201.png" alt="image-20241111205105201" style="zoom:67%;" />

### 2.1.1 k-mins MinHash Sketch

<img src="pictures\image-20241111205629009.png" alt="image-20241111205629009" style="zoom: 50%;" />

<img src="pictures\image-20241111205651568.png" alt="image-20241111205651568" style="zoom: 67%;" />

### 2.1.2 k-partition MinHash Sketch

<img src="pictures\image-20241111210042198.png" alt="image-20241111210042198" style="zoom: 50%;" />

<img src="pictures\image-20241111210112362.png" alt="image-20241111210112362" style="zoom:67%;" />

### 2.1.3 Bottom- k Min-Hash Sketch

<img src="pictures\image-20241111211107949.png" alt="image-20241111211107949" style="zoom:50%;" />

<img src="C:\Users\LX\AppData\Roaming\Typora\typora-user-images\image-20241111211155717.png" alt="image-20241111211155717" style="zoom: 67%;" />

### 2.1.4 Composability of MinHash Sketches

<img src="pictures\image-20241111215549803.png" alt="image-20241111215549803" style="zoom: 33%;" />

<img src="pictures\image-20241113155502721.png" alt="image-20241113155502721" style="zoom: 80%;" />

<img src="pictures\image-20241113155547296.png" alt="image-20241113155547296" style="zoom: 80%;" />

<img src="pictures\image-20241113155616680.png" alt="image-20241113155616680" style="zoom:80%;" />

## 2.2 Reservoir Sampling

<img src="pictures\image-20241113192041924.png" alt="image-20241113192041924" style="zoom: 50%;" />

<img src="pictures\image-20241113192125543.png" alt="image-20241113192125543" style="zoom:67%;" />

<img src="pictures\image-20241113192215542.png" alt="image-20241113192215542" style="zoom:67%;" />

<img src="pictures\image-20241113192247114.png" alt="image-20241113192247114" style="zoom:67%;" />

<img src="C:\Users\LX\AppData\Roaming\Typora\typora-user-images\image-20241113195505574.png" alt="image-20241113195505574" style="zoom: 50%;" />

<img src="pictures\image-20241113200340197.png" alt="image-20241113200340197" style="zoom:67%;" />

## 2.3 MinHash Sampling

### 2.3.1 方法

<img src="pictures\image-20241113193150283.png" alt="image-20241113193150283" style="zoom:67%;" />

### 2.3.2 应用

<img src="pictures\image-20241113193406338.png" alt="image-20241113193406338" style="zoom:67%;" />

<img src="pictures\image-20241113193434720.png" alt="image-20241113193434720" style="zoom:67%;" />

<img src="pictures\image-20241113193522036.png" alt="image-20241113193522036" style="zoom:67%;" />

**k-mins sketch**：适用于需要**高精度的相似性检测**，适合数据量相对中等且计算资源充足的场景。

**Bottom-k sketch**：适合**大规模、实时性要求高的场景**，具有更好的空间效率和计算效率。

**k-partition sketch**：适用于**分布式数据或需要分区处理的场景**，通过分区采样在分布式环境下获得更均衡的样本。

# 3. 题目

## 3.1 指数分布

<img src="C:\Users\LX\Desktop\大数据应用算法\pictures\image-20241113141321181.png" alt="image-20241113141321181" style="zoom:67%;" />

<img src="pictures\image-20241113141358812.png" alt="image-20241113141358812" style="zoom: 50%;" />

## 3.2 可合并的Sketch算法

**请简要回答为什么需要设计可合并的 Sketch 算法？可合并的 Sketch 算法主要是用于什么场景？**

可合并的 Sketch 算法是为了解决**大规模数据流处理和分布式系统中的近似查询问题而设计的**。这些算法能够对数据流进行压缩和摘要，以便在有限的内存和有限的通信带宽条件下处理大量的数据。
可合并的 Sketch 算法主要用于以下场景：
**数据流处理**：在数据流处理系统中，数据以高速率持续到达，无法全部存储在内存中。可合并的 Sketch 算法通过在有限的内存中维护摘要信息，例如频率估计、矩估计等，能够对数据流进行实时查询和分析。
**分布式系统：**在分布式系统中，数据通常分布在多个节点上，而节点之间的通信成本较高。可合并的 Sketch 算法允许在分布式环境下对数据进行分布式计算和聚合，从而减少数据传输量和通信开销。
**网络监测和流量分析：**可合并的 Sketch 算法可以用于网络监测和流量分析，例如统计网络中不同类型的流量、识别网络中的异常行为或研究网络拓扑结构等。

## 3.3 MG算法

**给定数据流 D=(1,2,5,1,4,2,3,3,2,4,5,2)，假设 k=3，请详细描述 Misra‐Gries 算法在该数据流上的运行步骤。**

```
T=0: MG{}
T=1:输入1，MG{1:1}
T=2:输入2，MG{1:1,2:1}
T=3:输入5，MG{1:1,2:1,5:1}
T=4:输入1，MG{1:2,2:1,5:1}
T=5:输入4，MG{1:1}
T=6:输入2，MG{1:1,2:1}
T=7:输入3，MG{1:1,2:1,3:1}
T=8:输入3，MG{1:1,2:1,3:2}
T=9:输入2，MG{1:1,2:2,3:2}
T=10:输入4，MG{2:1,3:1}
T=11:输入5，MG{2:1,3:1,5:1}
T=12:输入2，MG{2:2,3:1,5:1}
```

## 3.4 Morris counter

**请解释 Morris 计数算法的基本原理？它为什么能够做到只用 O(loglogn)的空间来对 n 个数据进行计数？**

Morris计数器的原理简单来说是记录一个计数器s，每次以
$$
2^{-s}
$$
的概率给s+1，最终返回的估计结果是
$$
2^s-1
$$
。对于最终的估计量来说，我们所记录的s是估计量的log结果；而当我将s这个结果存储到内存中时，我还需要再进行一步log得出我最终所需的**位数**，所以最后的空间只需要loglogn。

## 3.5 Minhash

**MinHash 算法有哪三种实现方式？它们各自有哪些优缺点？**

Minhash算法的三种实现方式分别是：k-mins sketch、Bottom-k sketch、k-partition sketch。

**K-mins**算法是使用k个hash函数，每个哈希函数都对序列求一遍哈希，取出每个哈希函数求出的最小值代表该序列的最小哈希，优点是结果可信度高，缺点是**浪费空间**；

**Bottom-k**是只用一个哈希函数，对序列求哈希，取出前k个最小的代表该序列的最小哈希，优点是操作简单，节省空间；**结果精度较低，容易受哈希冲突影响。**

**k-partition**综合上述两种方法，先分成k个部分然后再用一个函数做哈希，取出每个部分中最小的哈希值，优点是结果可信度比较好，也比较节省空间，缺点是**操作繁琐**，大批量数据很麻烦。

![image-20241113150103795](pictures\image-20241113150103795.png)

## 3.6 K-mins Minhash

**给定一个包含 n 个不同元素的集合 A，请证明在采用 k‐mins MinHash 算法来 构造集合 A 的 k‐mins sketch 的过程中，sketch 发生更新的期望次数为 O(k lnn)。**

假设我们已经有一个包含 k个最小哈希值的 sketch，并且我们正在插入第 i+1 个元素的哈希值。考虑新哈希值 
$$
h_{i+1}
$$
会更新现有的 k个最小值的概率。

- 如果 
  $$
  h_{i+1}
  $$
  小于当前的第 j 个最小哈希值，那么它将替换第 j个最小哈希值。

- 假设当前 k 个最小哈希值是 
  $$
  h_1,h_2,...,h_k
  $$
  ，这些哈希值的排序是从小到大的。如果 
  $$
  h_{i+1}
  $$
  是从新的元素的哈希值中抽取的，那么
  $$
  h_{i+1}
  $$
  小于当前最小哈希值的概率是 
  $$
  \frac{k}{i+1}
  $$
  ，因为在所有前 i个哈希值中，选择 k 个最小哈希值的期望概率是 
  $$
  \frac{k}{i+1}
  $$
  （每个新的哈希值有 k个位置可能被它替换）。

因此，每个新的哈希值更新 k个最小值的期望次数是 
$$
\frac{k}{i+1}
$$
假设我们有 n个元素，总共有 n 次插入操作，每次插入操作的期望更新次数为 
$$
\frac{k}{i+1}
$$
因此，总的期望更新次数为：
$$
\mathbb{E}[\text{总更新次数}] =  \sum_{i=1}^n \frac{k}{i}
$$

```
\mathbb{E} 用于表示期望值符号。
\text{总更新次数} 用于在期望符号中显示文本“总更新次数”。
\sum_{i=1}^{n} \frac{k}{i} 表示从 i=1 到 n 的求和，每一项为 k / i
```

这个求和式是一个调和级数，已知调和级数的上界为 ln⁡n，即：
$$
\because \sum_{i=1}^n \frac{1}{i} = O(lnn) \\
\therefore \mathbb{E}[\text{总更新次数}]=O(klnn)
$$

## 3.7 k-mins Minhash

**给定集合 A={a, b, c, d, e, h}, B={a, c, e, f, g, m, n}。假设采用 k‐mins MinHash 算 法来处理集合 A 与 B。令 k=4，得到集合 A 和集合 B 的 k‐mins sketch 分别为 S(A)=( 0.22, 0.11, 0.14, 0.22), S(B)=( 0.18, 0.24, 0.14, 0.35)**

1. **请计算 A 和 B 的 Jaccard 相似性。**
2. **请根据 S(A)和 S(B)来计算集合 A 与 B 合并后的 k‐mins sketch，即 S(A∪B)。**
3. **请基于 S(A∪B)估计集合 A 和集合 B 的 Jaccard 相似性。**
4. **在该问题中，基于 k‐mins MinHash 算法的估计方差为多少？**

<img src="pictures\image-20241113160001258.png" alt="image-20241113160001258" style="zoom: 67%;" />

<img src="C:\Users\LX\Desktop\大数据应用算法\pictures\image-20241113160036308.png" alt="image-20241113160036308" style="zoom:67%;" />

<img src="pictures\image-20241113160105006.png" alt="image-20241113160105006" style="zoom:67%;" />

<img src="pictures\image-20241113160134650.png" alt="image-20241113160134650" style="zoom:67%;" />

## 3.8 k-mins Minhash/图

**给定一个有向图 G=(V, E)，以及一个节点 v。在图 G 中可以到达节点 v 的集合定义为 Reach‐1(v)={u∈V|u 在图 G 中可达 v}。对于集合 Reach‐1(v)，我们可以采用 k‐mins MinHash 算法来生成该集合的一个 k‐mins sketch，记为 S(v)。**

**1. 请设计一个算法计算图中所有节点的 k‐mins sketch，即对于任意的 v 计算所有的 S(v)。**

<img src="pictures\image-20241113160706149.png" alt="image-20241113160706149" style="zoom:67%;" />

<img src="pictures\image-20241113161010084.png" alt="image-20241113161010084" style="zoom:67%;" />

**2. 请描述如何用 S(u)和 S(v)来计算| Reach‐1(v)∪Reach‐1(u)|?**

<img src="pictures\image-20241113161712990.png" alt="image-20241113161712990" style="zoom: 33%;" />

<img src="pictures\image-20241113161156659.png" alt="image-20241113161156659" style="zoom:67%;" />

## 3.9 bloom-filter

**给定一个集合D,假设你已经构造了一个长度m=2b比特的bloom filter。你的同学也想用你构造的这个bloom filter来处理“元素x是否在D中”的查询，但是你同学只有b比特的空间，请问他该如何实现他的目标？注意：假设我们不允许他重新针对集合D构造一个长度为b的bloom filter,而是要求在你构造的长度为2b的bloom filter的基础上去实现。**

​	通过将原始长度为 2b2b2b 的 Bloom Filter 划分为两部分，其中 **前 bbb 比特** 用于查询，**后 bbb 比特** 保留不使用，我们能够在同学只有 bbb 比特的情况下实现查询元素是否在集合 DDD 中的目标。这个方法的关键在于 **保持哈希函数一致性**，确保查询时能准确地映射到 Bloom Filter 的前 bbb 个比特，进而进行有效的查询。

