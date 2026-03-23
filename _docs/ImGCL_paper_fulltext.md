# ImGCL: Revisiting Graph Contrastive Learning on Imbalanced Node Classification (NeurIPS 2023)

**Source**: (2023 NeurIPS) ImGCL - Revisiting Graph Contrastive Learning on Imbalanced Node Classification.pdf


---
## Page 1

ImGCL: Revisiting Graph Contrastive Learning on
Imbalanced Node Classification
Liang Zeng1, Lanqing Li2*, Ziqi Gao3, Peilin Zhao2, Jian Li1*
1Institute for Interdisciplinary Information Sciences (IIIS), Tsinghua University
2Tencent AI Lab
3Hong Kong University of Science and Technology
zengl18@mails.tsinghua.edu.cn, {lanqingli, masonzhao}@tencent.com,
zgaoat@connect.ust.hk, lijian83@mail.tsinghua.edu.cn
Abstract
Graph contrastive learning (GCL) has attracted a surge
of attention due to its superior performance for learning
node/graph representations without labels. However, in prac-
tice, the underlying class distribution of unlabeled nodes for
the given graph is usually imbalanced. This highly imbal-
anced class distribution inevitably deteriorates the quality of
learned node representations in GCL. Indeed, we empirically
find that most state-of-the-art GCL methods cannot obtain
discriminative representations and exhibit poor performance
on imbalanced node classification. Motivated by this observa-
tion, we propose a principled GCL framework on Imbalanced
node classification (ImGCL), which automatically and adap-
tively balances the representations learned from GCL without
labels. Specifically, we first introduce the online clustering
based progressively balanced sampling (PBS) method with
theoretical rationale, which balances the training sets based
on pseudo-labels obtained from learned representations in
GCL. We then develop the node centrality based PBS method
to better preserve the intrinsic structure of graphs, by up-
weighting the important nodes of the given graph. Extensive
experiments on multiple imbalanced graph datasets and im-
balanced settings demonstrate the effectiveness of our pro-
posed framework, which significantly improves the perfor-
mance of the recent state-of-the-art GCL methods. Further
experimental ablations and analyses show that the ImGCL
framework consistently improves the representation quality
of nodes in under-represented (tail) classes.
1
Introduction
Recently, graph contrastive learning (GCL) has become
the de facto standard for self-supervised learning on
graphs (Zhu et al. 2021a) due to its superior performance
as compared to the supervised counterparts. (Thakoor et al.
2022; Bielak, Kajdanowicz, and Chawla 2021; You et al.
2020; Zhu et al. 2021b). Inheriting the advantage of self-
supervised learning, GCL frees the model from the reliance
on label information in the graph domain, where labels can
be costly and error-prone in practice (Yang and Xu 2020)
while unlabeled/partially labeled data is prevalent, such as
fraudulent user detection (Kumar et al. 2018) and molecular
property prediction (Ma et al. 2020). Typically, most GCL
*Corresponding authors.
Copyright © 2023, Association for the Advancement of Artificial
Intelligence (www.aaai.org). All rights reserved.
Figure 1: (a) Class distribution of the Amazon-Computers
dataset sorted in decreasing order, where pre-training sets
are highly imbalanced but testing sets are balanced. (b)
Compared to the GBT (Bielak, Kajdanowicz, and Chawla
2021) baseline model on Amazon-computers, ImGCL+GBT
substantially outperforms GBT on the overall accuracy.
On three splits (head/middle/tail) depicted in different col-
ors (blue/yellow/red), ImGCL+GBT improves the perfor-
mance on middle and tail classes by a large margin with a
slight sacrifice of accuracy on head classes.
methods first construct multiple graph views via stochastic
augmentation functions on the input graph and then learn
discriminative representations by maximizing the represen-
tation consistency between two views.
Despite the prevalence and effectiveness of such methods,
existing GCL methods mostly assume the datasets are care-
fully curated and well balanced across classes. However, un-
labeled graph data randomly gathered in the wild often ex-
hibits a highly imbalanced class distribution (Barab´asi and
Albert 1999; Liu, Nguyen, and Fang 2021) and thus im-
plicitly deteriorates the quality of learned representations in
GCL methods (Jiang et al. 2021). For instance, Fig. 1(a)
illustrates the class distribution sorted in decreasing order
of the Amazon-Computers dataset (Shchur et al. 2018),
a network of co-purchase goods containing 10 classes.
In the pre-training set, we can clearly find that a small
fraction of classes take up massive samples (a.k.a., head
classes) and the rest of classes are assigned only a few sam-
ples (a.k.a., tail classes). Note that the highly imbalanced
property of label information within the training data is un-
known to the GCL methods. This important property im-
plicitly exists and is largely ignored by the current GCL
methods (Zhu et al. 2021a). However, to impose a fair eval-
The Thirty-Seventh AAAI Conference on Artificial Intelligence (AAAI-23)
11138


---
## Page 2

uation metric, on imbalanced node classification, the test-
ing set is balanced across all classes. Shown as the blue
line of Fig. 1(b), we adopt one of the state-of-the-art GCL
methods—GBT (Bielak, Kajdanowicz, and Chawla 2021)—
to conduct experiments on imbalanced node classification.
We can find that the baseline GBT model obtains very poor
results, especially on the under-represented middle and tail
classes, which naturally spurs a question:
How to improve the representation learning of GCL on
highly imbalanced node classification?
Recent works related to this question (He et al. 2021;
Kang et al. 2020a) explore balanced feature spaces to learn
powerful representations not just for head classes but also for
tailed classes. Kang et al. (2020b) introduce PBS method to
innovate imbalanced representation learning, achieving re-
markable success by decoupling the learning procedure into
a representation learning stage and a classification stage.
However, this method is impractical for the GCL setting
since it requires knowing labels. This dilemma motivates us
to explore how to implicitly obtain label information to im-
prove node representations in the traditional GCL setting.
In this paper, we present a principled GCL framework on
Imbalanced node classification (ImGCL), to automatically
and adaptively balance the representations learned from
GCL without ground truth labels. To perform class-balanced
re-sampling, ImGCL obtains pseudo-labels via online clus-
tering of learned representations in GCL. Moreover, we pro-
pose the node centrality based PBS method tailored for the
graph domain, which assigns a higher probability to retain
nodes with high node centrality scores when down-sampling
the head class nodes in online clustering based PBS. This
scheme is able to guide the model to learn node represen-
tations with higher node centrality scores, which is consid-
ered more important by having abundant structural connec-
tivity when performing message passing. We also provide
theoretical insight into the PBS method. Furthermore, ex-
isting GCL models can be seamlessly incorporated into the
proposed ImGCL framework in a plug-and-play manner. In
short, our main contributions can be summarized as follows:
• New problem and insights: we introduce a practically im-
portant but under-explored problem, namely graph con-
trastive learning on imbalanced node classification. We
empirically identify that the recently proposed GCL meth-
ods are vulnerable to node class imbalance and result in
large performance degradation.
• New principled framework: we propose a novel ImGCL
framework, which utilizes the node centrality based PBS
method. ImGCL automatically and adaptively balances
the representations learned from GCL without knowing
labels. Moreover, existing GCL models can be seamlessly
incorporated into our framework.
• Convincing empirical results: we conduct comprehen-
sive experiments to show that the ImGCL framework
achieves superior performance compared with the re-
cently proposed GCL methods on imbalanced node classi-
fication. Extensive ablation studies also demonstrate that
the ImGCL framework improves the representations of
the under-represented (middle and tail) classes.
2
Background
Let G = (V, E, X) denote a graph, where V = {vi}N
i=1
is a set of N nodes, and E ⊆V × V is a set of edges
between nodes. X = [x1, x2, · · · , xN]T ∈RN×d repre-
sents the node feature matrix and xi ∈Rd is the feature
vector of node vi, where d is the feature dimension. The ad-
jacency matrix A ∈{0, 1}N×N is defined by Ai,j = 1
if (vi, vj) ∈E and 0 otherwise. More detailed discussions
about related works can be found in the appendix.
Graph Contrastive Learning (GCL).
Given an input
graph, GCL aims to learn effective graph/node representa-
tions that can be transferred to downstream tasks by con-
structing positive and negative sample pairs (Thakoor et al.
2022; Bielak, Kajdanowicz, and Chawla 2021; Zhang et al.
2021a; Xu et al. 2021; Veliˇckovi´c et al. 2018; Sun et al.
2020; Hassani and Khasahmadi 2020). Specifically, we uti-
lize two augmentation functions t1, t2
∼
T to gener-
ate graph views eG1 = (f
X1, e
A1) = t1(G) and eG2 =
(f
X2, e
A2) = t2(G), where T is the set of graph augmen-
tation functions, such as node dropping, edge perturbation,
and subgraph sampling (Zhu et al. 2021b). We then obtain
node representations for the two graph views via the (param-
eter shared) GNN encoder f(·), denoted by Z = f(f
X1, e
A1)
and Z′ = f(f
X2, e
A2) respectively. Given the latent repre-
sentations, we optimize the parameters of the GNN encoder
by a pre-defined contrastive loss. For any node u in eG1, we
aim to score the positive pairs (u, u+) higher compared to
other negative pairs (u, u−). Typically, the negative samples
u−are sampled from other nodes of the augmented graph
views eG2 in the same batch. The commonly-used InfoNCE
loss (Chen et al. 2020) can be defined as:
LNCE(u) = −log
s (zu, zu+, τ)
s (zu, zu+, τ) + P
u−̸=u s (zu, zu−, τ),
(1)
where s (zu, zu+, τ) indicates the similarity between node
representations of positive pairs, while s (zu, zu−, τ) is the
similarity between negative pairs. s is a contrasting function
to measure the similarity between two node representations,
which is typically defined as: s(zu, zv, τ) = exp(zu·zv/τ).
τ represents the temperature hyper-parameter.
Imbalanced Learning.
Imbalanced learning seeks to
learn a model from the training set with an imbalanced class
distribution, where head classes take up the vast majority of
samples and tail classes occupy only a few samples, and gen-
eralize well on a balanced testing set (Kang et al. 2020b;
Zhang et al. 2021b,c; Liu et al. 2019; Li et al. 2022). For
a K-way node classification problem, let {vi, yi}N
i=1 be an
imbalanced training set. The total number of the training set
over K classes is N = PK
k=1 Nk, where Nk denotes the
number of samples in class k. Let π be the vector of label
frequencies, where πk = Nk/N denotes the label frequency
of class k. Without loss of generality, we assume that the
classes are sorted by πk in a descending order (i.e., if the
class index i < j, then Ni ≥Nj, and N1 ≫NK). We de-
note by N1/NK the imbalance ratio of the dataset. Conven-
11139


---
## Page 3

GNN
Encoder
Epoch = 1
Epoch = T
.
.
.
Embedding space
Embedding space
Unlabeled
Nodes
Pseudo
Labels
Vector
(a)
(b)
Imbalanced
Pseudo-labels
Balanced
Pseudo-labels
1
1
0
1
1
0
Masked vector
Graph 
Contrastive 
Loss
Downstream
Task
Down-
sampling
Down-
sampling
Node centrality
Masked
Vector
Graph 
Contrastive 
Loss
...
...
...
Node embeddings
Figure 2: Overview of the proposed ImGCL framework. (a) Graph contrastive learning (GCL) methods take the graph as
input and produce the embeddings of each node. (b) Node centrality based progressively balanced sampling (PBS) method
automatically and adaptively balances the representations learned from GCL without knowing the labels.
tionally, evaluations on imbalanced learning report statistics
for the head, middle, tail, and overall classes separately.
3
ImGCL: The Proposed Framework
In this section, we first introduce the progressively balanced
sampling (PBS) method, which is impractical in our self-
supervised setting since it requires knowing the labels. Mo-
tivated by the property of GCL, we generate pseudo-labels
by clustering the node representations. Finally, we utilize the
proposed node centrality based PBS method to adaptively
attend to ’important’ nodes of the given graph during the
down-sampling phase. The overall framework of ImGCL is
shown in Fig. 2.
3.1
Progressively Balanced Sampling (PBS)
Sampling Strategies.
As suggested in (He et al. 2021), the
probability pk of sampling a node for the given graph from
the class k is defined as:
pk =
N q
k
PK
i=1 N q
i
,
(2)
where q ∈[0, 1]. Different sampling strategies have different
specific values of q.
PBS (Kang et al. 2020b).
In imbalanced learning, the test-
ing dataset is balanced whereas the training dataset is highly
imbalanced. A single data sampling strategy which fits only
one case, the balanced dataset (pM
k
=
1
K by setting q = 0
in Eq. 2) or the imbalanced dataset (pR
k =
Nk
PK
i=1 Ni by set-
ting q = 1 in Eq. 2), cannot account for the class distri-
bution shift between the training and testing datasets. Thus,
in order to learn high-quality representations from the im-
balanced dataset, we adopt two data samplers with adaptive
sampling strategies known as decoupled training in long-
tailed learning literature (Zhang et al. 2021b). Formally, at
training step t, data are sampled according to a linear com-
bination of the random and mean strategies, controlled by a
parameter α ∈[0, 1]. Therefore, the probability of sampling
a node from the class k is given by:
pPB
k = α ∗pR
k + (1 −α) ∗pM
k
= α ∗
Nk
PK
i=1 Ni
+ (1 −α) ∗1
K .
(3)
Intuitively, at early stages of the training phase, an imbal-
anced class distribution is used for representation learning
of the feature extractor. At later stages, the model benefits
more from a balanced dataset for training an unbiased classi-
fier. Therefore, the control parameter α should progressively
decrease from 1 to 0 during the training phase. Concretely,
at training step t, α is calculated by (Kang et al. 2020b):
α = 1 −t
T , where T is the total number of training epochs.
3.2
Online Clustering Based PBS
The PBS method requires real labels to adjust the class dis-
tribution during training, which compromises its practical-
ity. Since GCL is typically applied for the label-free sce-
nario, we cannot directly apply PBS here. On one hand,
(McPherson, Smith-Lovin, and Cook 2001) have revealed
the homophily phenomenon in homophilic graphs, i.e., the
nodes with similar features tend to be connected with each
other and share the same label. On the other hand, the mo-
tivation of GCL is to learn representations in which simi-
lar node pairs stay close to each other while dissimilar ones
are far apart. We propose to connect these two facts by the
pseudo-label method in GCL (Caron et al. 2018), which iter-
atively generates artificial labels by the model itself to make
the PBS method applicable in the label-free GCL scenarios.
We utilize the emergence of representation clusters learned
from GCL to generate pseudo-labels of each node and then
apply a down-sampling strategy to improve the quality of
node representations in middle and tail classes.
Specifically, suppose there are K-classes for the node
classification task. At the certain iteration t of the training
phase, we obtain the node representation Zt ∈RN×D via
the learned GNN encoder, where D is the hidden dimension.
11140


---
## Page 4

We apply the clustering algorithm to the nodes in the embed-
ding space to produce a set of K prototypes {c1, . . . , cK}.
Formally, we intend to learn a D ×K centroid matrix C and
a one-hot cluster assignment vector ˆyn ∈RK
+ for each node
n of the given graph by solving the following problem:
min
C∈RD×K
1
N
N
X
n=1
min
ˆyn ∥zt,n −Cˆyn∥2
2 such that
ˆy⊤
n 1K = 1,
(4)
where zt,n ∈RD denotes the n-th node embedding vec-
tor of Zt, and 1K ∈RK is the vector with all elements of
1. Solving this above problem provides a set of optimal as-
signments {ˆy∗
n|n = 1, . . . , N} and a centroid matrix C∗.
These assignments are then used as pseudo-labels. Note that
We set the number of centroids (clusters) K in the cluster-
ing algorithm equal to the number of classes in the training
dataset, which is an input hyperparameter of the classifica-
tion task. We also perform the hyperparameter study in Ap-
pendix D and find that it is generally a good choice and pre-
vents performance fluctuation. In order to avoid trivial solu-
tions and empty clusters, we use the constrained K-means
clustering (Bradley, Bennett, and Demiriz 2000) to instanti-
ate the clustering algorithm. It can implement the K-means
clustering algorithm whereby a minimum size for each clus-
ter can be specified. Thus, we can address the representation
collapse problem (Fang et al. 2021) which would produce a
highly imbalanced pseudo-label distribution.
Theoretical Analysis.
In order to justify the PBS method,
we theoretically prove that, the classifier learned iteratively
by balanced sampling with pseudo-labels on the imbalanced
dataset can converge to the optimal balanced classifier with
a linear rate. Detailed proofs can be found in Appendix B.
Consider a binary classification problem with two Gaus-
sian distributions with different means and the equal vari-
ance. Suppose the data generating distribution is PXY and
the probability of positive labels (+1) and negative labels
(−1) are PY (1) and PY (−1), respectively. We have X|Y =
+1 ∼N(µ1, σ2) conditioned on Y = +1 and similarly,
X|Y
= −1 ∼N(µ2, σ2) conditioned on Y
= −1. In
addition, suppose µ1 < µ2 without loss of generality. It is
straightforward to verify that (Bishop and Nasrabadi 2006)
the optimal decision boundary of a balanced Bayes classi-
fier is θ∗≡
µ1+µ2
2
. At the first iteration t = 1, we start
from the imbalanced unlabeled dataset and generate pseudo-
labels ˆY0 by the clustering method. Then, we obtain the esti-
mated decision boundary ˆθ1 = ˆµ1+ˆµ2
2
and produce pseudo-
labels ˆY1. The fact that the initial dataset being imbalanced
(PY (1) ̸= PY (−1)) leads to biased ˆθ1. We iteratively obtain
the estimator ˆθt and pseudo-labels ˆYt when t ≥2.
Proposition 1. Consider the above setup. Suppose there is a
(sufficiently large) integer T such that |ˆθT −θ∗| ≪|µ2−µ1|,
|(ˆθT −θ∗)(µ2 −µ1)| ≪σ2. Our estimator ˆθT converges
to the optimal balanced decision boundary θ∗, i.e., |ˆθt+1 −
θ∗| ≤C · |ˆθt −θ∗|, ∀t ≥T with a linear convergence rate
C = 2
π < 1.
Interpretation.
In the context of PBS, the classifier is
trained starting from the imbalanced data distribution. In-
tuitively, by iteratively down-sampling head classes, the
data distribution gradually becomes balanced, on which the
trained classifier also converges to the balanced optimum.
3.3
Node Centrality Based PBS
To further improve the representations of nodes in under-
represented (middle and tail) classes, we incorporate graph
structural information when performing the node central-
ity based PBS method. In network science, node centrality,
which measures how important a node is in a graph, is an
important metric to understand the influence of each node of
a graph (Barab´asi 2013). Therefore, we propose an adaptive
down-sampling scheme based on node centrality to balance
the class distribution over all classes. For each node class,
we sample nodes of higher centrality with higher probability
to better preserve the intrinsic structures of graphs in learned
representations. We herein utilize the PageRank centrality
due to its simplicity and effectiveness (Barab´asi 2013). For-
mally, the centrality values are calculated by the iterative
form: σ = αAD−1 + 1, where σ ∈RN is the PageR-
ank centrality score vector for each node, α is a damping
factor to control the probability of randomly jumping to an-
other node in the graph, A and D denote the adjacency and
the degree matrix of the input graph respectively, and 1 is
the all-ones identity vector. Note that we pre-calculate the
PageRank score σ before the training phase of GCL. When
performing down-sampling of the head classes, we calculate
the probability of each node based on σ. Formally, for node
v in class j with the centrality score σv, the probability with
the normalized centrality score is defined as:
pNPB
v,j = max
 σv −σmin
σmax −σmin
· pPB
j , pτ

,
(5)
where pPB
j
is the progressively balanced sampling probabil-
ity of class j, σmax and σmin are the maximum and mini-
mum value of the centrality score, and pτ is a cut-off prob-
ability to ensure that nodes with extremely low probabili-
ties can also be sampled. In node centrality based PBS, we
then perform a normalization step that transforms pNPB
v,j into
probabilities and then use it to balance the class distribution.
Concretely, we select certain nodes of the original graph in
form of a masked vector m ∈RN by sampling each node in-
dependently according to pNPB
v,j . We then calculate the graph
contrastive loss only on these selected node representations,
as shown in Fig. 2.
3.4
Learning Framework
ImGCL is a general GCL framework for imbalanced node
classification, which can be readily applied with existing
GCL methods adopting the two-branch design (Thakoor
et al. 2022; Bielak, Kajdanowicz, and Chawla 2021; You
et al. 2020). ImGCL does not rely on specific approaches
of graph view augmentation, graph view encoding, or rep-
resentation contrasting in GCL. In ImGCL, we set the num-
ber of clusters K in the node centrality based PBS method
equal to the number of classes in the downstream task. One
11141


---
## Page 5

Tail class
(a) GBT
(c) ImGCL+GBT
(b) GBT+PBS
Head
Tail
Figure 3: Visualization of the testing set on Amazon-Computers. Each point in the figure is colored by real labels.
challenge of the implementation is that the quality of the
generated pseudo-labels and representations are mutually-
dependent, which could destabilize the training loop. In re-
sponse, we re-balance the class distribution every B epochs,
thus there are T/B times to adjust the class distribution.
In order to improve the representations of nodes in under-
represented (middle and tail) classes, we select N × l nodes
during the pre-training phase in ImGCL, where l = 10%
equals the ratio of training data in the down-stream task.
We down-sample the head nodes to the required num-
ber according to their PageRank centrality scores as intro-
duced in Sec. 3.3. Following the linear evaluation scheme
of GCL (Zhu et al. 2021b), we train a linear classifier on
the balanced dataset with 10% randomly selected nodes af-
ter obtaining node representations. The training algorithm of
ImGCL is summarized as follows.
Algorithm 1: The ImGCL pre-training algorithm
Input: The input graph G, GNN encoder F.
Parameter: Number of nodes N, number of clusters
K, re-balanced labeling frequency B,
the ratio of selected nodes l.
Output: Pre-trained GNN encoder F.
1 Calculate the node centrality vector σ of G.
2 for epoch = 0, 1, 2, · · · do
3
Draw two augmentation functions t1 ∼T ,
t2 ∼T .
4
Generate two graph views eG1 = t1(G) and
eG2 = t2(G).
5
Obtain node representations U of eG1 and V of eG2
using the GNN encoder F.
6
if epoch mod B == 0 then
7
Cluster node representations to obtain
pseudo-labels.
8
Calculate the normalized centrality score
pNPB with Eq. 5.
9
Obtain the masked vector m according to
pNPB, which satisfies ∥m∥= N × l.
10
Compute the contrastive object L on these
selected node representations U ⊙m and
V ⊙m with Eq. 1.
11
Update the parameters of F with L.
4
Case Study: Learning Discriminative
Representations on Amazon-Computers
We are particularly interested in the learned representa-
tions among different methods. For a more intuitive com-
parison and to further demonstrate the effectiveness of the
ImGCL framework, we design experiments of visualization
on Amazon-computers. We utilize the output representations
on the last layer of vanilla GBT, GBT with PBS to train
the classifier on training data but without node centrality
based PBS on representation learning in GCL, and GBT
with ImGCL. We plot the learned representations of the test-
ing graph dataset using t-SNE (Van der Maaten and Hin-
ton 2008). As shown in Fig. 3, vanilla GBT clearly exhibits
the minority collapse (Fang et al. 2021) phenomenon, and
the representations in tail classes are mixed together, which
fundamentally limits the performance in the tail classes. In
Fig. 3 (b), we can find clearer boundaries among different
classes but the representations in tail classes are still mixed
together, which suggests that it is important to explore bal-
anced representation spaces in GCL methods. In compari-
son, the learned representations of GBT+ImGCL (Fig. 3 (c))
have distinct boundaries among different classes and more
compact intra-class structures, which highlights the effec-
tiveness of our proposed ImGCL framework.
5
Experiments
In this section, we provide empirical results to demonstrate
the effectiveness of our ImGCL framework. We conduct ex-
tensive experiments on imbalanced graph datasets to mainly
answer the following questions: 1 (1) Can ImGCL generally
improve the performance of GCL methods on imbalanced
node classification? (Sec. 5.1) (2) How does ImGCL per-
form on different imbalanced types? (Sec. 5.2) (3) How does
ImGCL help improve GCL methods on imbalanced node
classification? (Sec. 5.2)
Dataset.
We use four widely-used datasets including
Wiki-CS, Amazon-computers, Amazon-photo, and DBLP,
to comprehensively study the performance of transductive
node classification. In order to validate the effectiveness of
ImGCL on the imbalanced node classification setting, we
1Due to space limitations, ablations on different components in
ImGCL and hyperparameter studies are provided in the appendix.
11142


---
## Page 6

Category
Method
Available Data
Amazon-Computers
Amazon-Photo
Wiki-CS
DBLP
Raw features
X
33.99(4.47)
38.07(4.03)
34.50(2.06)
38.51(0.80)
Node2vec
A
69.80(2.69)
69.44(0.38)
51.76(2.24)
50.41(2.77)
DeepWalk
A
69.67(2.36)
69.00(0.62)
51.32(2.17)
50.57(2.88)
DeepWalk + features
X, A
70.20(3.30)
71.60(3.31)
51.51(2.51)
49.57(1.65)
GCL
DGI
X, A
10.88(2.09)
13.62(2.26)
17.11(4.83)
26.63(10.87)
MVGRL
X, A
13.40(3.01)
16.92(3.14)
45.97(2.42)
44.43(0.57)
InfoGraph
X, A
35.83(7.01)
53.57(13.29)
44.19(3.90)
48.26(5.95)
GRACE
X, A
41.54(2.51)
45.24(4.24)
54.20(3.97)
44.48(0.40)
BGRL
X, A
40.81(5.01)
51.18(10.49)
39.82(3.60)
49.58(3.99)
GBT
X, A
42.17(3.74)
60.73(3.64)
44.95(2.63)
58.51(5.30)
DGI
X, A
48.85(10.94)
47.99(9.03)
41.20(18.84)
50.39(9.17)
MVGRL
X, A
46.42(11.33)
50.86(9.49)
60.85(2.60)
51.90(7.42)
InfoGraph
X, A
75.44(5.30)
72.56(2.91)
68.96(5.86)
69.12(3.24)
ImGCL (ours)
GRACE
X, A
77.54(3.00)
68.89(4.41)
73.86(2.78)
63.61(4.91)
BGRL
X, A
67.82(10.24)
72.67(6.04)
59.35(15.44)
57.90(2.28)
GBT
X, A
78.62(1.73)
75.13(7.13)
73.08(12.45)
70.05(1.78)
Best ImGCL over GCL
39.61 ↑
34.47 ↑
28.13 ↑
23.76 ↑
Supervised
GCN
X, A, Y
46.83(1.52)
68.84(2.56)
59.84(2.02)
51.55(2.64)
GCN+PBS
X, A, Y
70.12(9.78)
73.34(8.28)
63.15(5.13)
73.11(2.76)
Table 1: Summary of accuracies (%) with standard deviation on imbalanced node classification. The ’Available Data’ means
data we can obtain for training, where X, A, and Y denote node features, the adjacency matrix, and labels respectively. We
highlight models in the ImGCL category with the gray background. The highest performance under each category is masked as
bold. The highest performance improvement of the GCL baseline w & w/o the ImGCL framework is underlined.
select an equal number of nodes in each class for the vali-
dation and testing dataset. Following (Zhu et al. 2021b), the
training set is randomly sampled from the rest according to
train/valid/test ratios = 1:1:8, which is highly imbalanced.
The descriptions, statistics, and the imbalance ratio of each
dataset can be found in Appendix H.
Evaluation Protocol.
For each experiment, we follow the
commonly-used linear evaluation scheme for GCL as intro-
duced in (Zhu et al. 2021b). The model is firstly trained in a
self-supervised manner, and then the learned representations
are used to train and test with a simple linear classifier. For
results in this section, we train each model in twenty runs
for different data splits and report the average performance
with the corresponding standard deviation for a fair compar-
ison. In what follows, we measure performance in terms of
accuracy, if not otherwise specified.
Imbalanced Types.
In order to comprehensively evaluate
the performance of ImGCL in different imbalanced types,
we introduce two imbalanced types (Jiang et al. 2021): Exp
and Pareto, parameterized by an imbalanced factor. Exp im-
balanced class distribution is given by an exponential func-
tion, where the higher imbalanced factor means the more im-
balanced graph. Pareto imbalanced class distribution is de-
termined by a Pareto distribution, where a lower imbalanced
factor means the smaller power value of a Pareto distribution
and thus the more imbalanced graph.
Baselines.
We consider representative baseline methods in
the following two categories: (1) traditional methods includ-
ing Node2vec (Grover and Leskovec 2016), DeepWalk (Per-
ozzi, Al-Rfou, and Skiena 2014), and raw features as in-
put without considering the graph topology. (2) deep learn-
ing methods including DGI (Veliˇckovi´c et al. 2018), MV-
GRL (Hassani and Khasahmadi 2020), InfoGraph (Sun et al.
2020), GRACE (Zhu et al. 2021b), BGRL (Thakoor et al.
2022), and GBT (Bielak, Kajdanowicz, and Chawla 2021).
We also directly compare ImGCL with the supervised coun-
terparts, i.e., the most representative model GCN (Kipf and
Welling 2016) and the variant of GCN trained with PBS.
Note that for all baselines, we report their performance on
the imbalanced experimental settings following their offi-
cial hyperparameters (detailed in Appendix H) based on the
PyGCL (Zhu et al. 2021a) open-source library.
5.1
Experimental Results on Node Classification
The empirical performance of imbalanced node classifica-
tion with the Exp type and 100 imbalanced factor is summa-
rized in Table 1. ImGCL consistently outperforms current
GCL baselines or even the supervised baselines. We summa-
rize our observations from the table as follows: (1) Recently
proposed GCL methods (Zhu et al. 2021a), which are eval-
uated on balanced testing sets, exhibit severe performance
degradation in our imbalanced node classification setting.
By incorporating our proposed ImGCL framework (the gray
background), these GCL methods improve by a large mar-
gin. Concretely, ImGCL+GCL achieves [39.61%, 34.47%,
28.13%, 23.76%] average absolute gain in accuracy than the
baseline GCL models on [Amazon-Computers, Amazon-
Photo, Wiki-CS, DBLP], respectively. We also find that the
recently proposed GBT (Bielak, Kajdanowicz, and Chawla
2021) obtains the best performance among a set of GCL
competitors. We think the reason is that feature decorrela-
11143


---
## Page 7

Type
Factor
Method
Catergory
Head Middle
Tail
All
Exp ↑
20
GBT
79.45
65.40
73.25 71.97
50
80.12
60.57
57.42 65.49
100
70.38
41.92
14.29 42.17
200
78.13
25.12
10.28 36.57
20
71.58
89.95
94.55 85.82
50
63.88
86.03
95.75 82.30
100
57.32
82.72
94.45 78.62
200
GBT+ImGCL
(ours)
46.95
75.27
76.42 67.12
Pareto ↓
2
GBT
83.31
61.01
52.58 65.17
1
82.06
57.22
39.85 59.46
2
53.81
70.25
93.19 72.20
1
GBT+ImGCL
(ours)
41.94
78.34
87.97 70.31
Table 2: Results of accuracy (%) on Amazon-computers
using the GBT baseline model under different imbalanced
types and factors. ↑means a higher imbalanced factor cor-
responds to a more imbalanced dataset. Instead, ↓means a
lower factor corresponds to a more imbalanced dataset.
tion method in GBT is more fit for the imbalanced node
classification and we adopt the GBT baseline model in the
following experimental analysis. (2) The traditional meth-
ods, e.g., Node2vec and DeepWalk, can achieve competitive
performance in the imbalanced node classification task com-
pared with GCL methods. We postulate the reason is that
the traditional network embedding methods can take advan-
tage of the homophily (Barab´asi 2013) property to utilize
the graph topology features which is important on imbal-
anced node classification, as reflected in (Liu, Nguyen, and
Fang 2021). Moreover, the “Raw features” method without
considering the graph topology cannot achieve satisfactory
performance, which indicates the necessity of utilizing the
graph topology features on imbalanced node classification.
(3) Compared with the supervised learning methods GCN
and GCN+PBS, the ImGCL framework achieves superior or
competitive performance on all datasets, which further cor-
roborates the effectiveness of our proposed framework.
5.2
Experimental Analysis
Imbalanced Type Analysis.
The performance for the
head, middle, and tail classes are usually reported on im-
balanced learning (Zhang et al. 2021b). We first divide the
training set of Amazon-computers into three disjoint groups
in terms of class size: {Head, Middle, Tail}. Head and Tail
each include the top and bottom 1/3 classes, respectively.
Because there are 10 classes of Amazon-computers in total,
the classes with sorted indices in decreasing order [1-3, 4-7,
8-10] belong to [Head (3 classes), Middle (4 classes), Tail (3
classes)] categories, respectively. To validate ImGCL across
different imbalanced class distributions, we design experi-
ments on Amazon-computers using the GBT baseline model
w & w/o the ImGCL framework. We consider two imbal-
anced types: Exp and Pareto. Four imbalanced factors [20,
50, 100, 200] are associated with Exp. Two imbalance fac-
tors [1, 2] are chosen for Pareto. In Table 2, we observe
that the more imbalanced dataset leads to the lower accuracy
(a) GBT
(b) ImGCL+GBT
Figure 4: Bias comparison between GBT models w & w/o
ImGCL on Amazon-Computers under imbalanced exper-
imental settings. Left: Per-class recall and precision w/o
ImGCL. Right: Per-class recall and precision with ImGCL.
The class index is sorted by the number of nodes in each
class in descending order. GBT w/o ImGCL clearly shows a
descending trend in recall while an ascending trend in preci-
sion. However, GBT with ImGCL has achieved a relatively
balanced recall and precision value over all classes.
of the model. Nevertheless, according to our assumption in
Proposition 1, the ImGCL+baseline model consistently out-
performs the baseline model.
Per-Class Analysis.
We analyze the per-class recall and
precision in Fig. 4 to better understand how ImGCL can help
improve GCL methods on the imbalanced node classifica-
tion. We test GBT on Amazon-computers under imbalanced
experimental settings introduced in Sec. 5. The GBT base-
line model exhibits highly skewed performance on head and
tail classes. The recall on the most majority and minority
class is 82.0% and 4.1% respectively, while the correspond-
ing numbers for precision are 30.9% and 95.8%. We observe
that GBT falsely classifies most of the tail class samples into
head classes with low confidence but have high confidence
on the correctly classified nodes in tail classes (Wei et al.
2021). In contrast, GBT with ImGCL obtains relatively bal-
anced recall and precision value on both head (the most ma-
jority precision: 30.9% →81.2%) and tail (the most minor-
ity recall: 4.1% →88.2%) classes, which leads to the sub-
stantial improvement in the overall accuracy (i.e., 42.17%
→78.62%) across all classes, as shown in Table 3.
6
Conclusion
In this work, we study how to improve the representations
of graph contrastive learning (GCL) methods on imbalanced
node classification, which is a very practical but rarely ex-
plored problem. We propose the principled ImGCL frame-
work, which automatically and adaptively balances the rep-
resentations learned from GCL without knowing labels and
then theoretically justifies it. Through extensive experiments
on multiple graph datasets and imbalance settings, we show
that ImGCL can significantly improve the recently proposed
GCL methods by improving the representations of nodes in
under-represented (tail) classes. For the future work, we will
explore more data types, such as bioinformatics graphs. We
hope our work will extend GCL to more realistic task set-
tings with (underlying) imbalanced node class distribution.
11144


---
## Page 8

Acknowledgments
Liang Zeng and Jian Li are supported in part by the National
Natural Science Foundation of China Grant 62161146004,
Turing AI Institute of Nanjing and Xi’an Institute for Inter-
disciplinary Information Core Technology.
References
Barab´asi, A.-L. 2013.
Network science.
Philosophical
Transactions of the Royal Society A: Mathematical, Phys-
ical and Engineering Sciences, 371(1987): 20120375.
Barab´asi, A.-L.; and Albert, R. 1999. Emergence of scaling
in random networks. science, 286(5439): 509–512.
Bielak, P.; Kajdanowicz, T.; and Chawla, N. V. 2021. Graph
Barlow Twins: A self-supervised representation learning
framework for graphs. arXiv preprint arXiv:2106.02466.
Bishop, C. M.; and Nasrabadi, N. M. 2006. Pattern recogni-
tion and machine learning, volume 4. Springer.
Bradley, P. S.; Bennett, K. P.; and Demiriz, A. 2000. Con-
strained k-means clustering. Microsoft Research, Redmond,
20(0): 0.
Caron, M.; Bojanowski, P.; Joulin, A.; and Douze, M. 2018.
Deep clustering for unsupervised learning of visual features.
In Proceedings of the European Conference on Computer
Vision (ECCV), 132–149.
Chen, T.; Kornblith, S.; Norouzi, M.; and Hinton, G. 2020.
A simple framework for contrastive learning of visual repre-
sentations. In International conference on machine learning
(ICML), 1597–1607. PMLR.
Fang, C.; He, H.; Long, Q.; and Su, W. J. 2021. Exploring
deep neural networks via layer-peeled model: Minority col-
lapse in imbalanced training. Proceedings of the National
Academy of Sciences, 118(43).
Grover, A.; and Leskovec, J. 2016. node2vec: Scalable fea-
ture learning for networks. In Proceedings of the 22nd ACM
SIGKDD international conference on Knowledge discovery
and data mining (KDD), 855–864.
Hassani, K.; and Khasahmadi, A. H. 2020.
Contrastive
multi-view representation learning on graphs.
In Inter-
national Conference on Machine Learning (ICML), 4116–
4126. PMLR.
He, J.; Kortylewski, A.; Yang, S.; Liu, S.; Yang, C.; Wang,
C.; and Yuille, A. 2021.
Rethinking Re-Sampling in
Imbalanced Semi-Supervised Learning.
arXiv preprint
arXiv:2106.00209.
Jiang, Z.; Chen, T.; Mortazavi, B.; and Wang, Z. 2021. Self-
Damaging Contrastive Learning. In International Confer-
ence on Machine Learning (ICML).
Kang, B.; Li, Y.; Xie, S.; Yuan, Z.; and Feng, J. 2020a. Ex-
ploring balanced feature spaces for representation learning.
In International Conference on Learning Representations
(ICLR).
Kang, B.; Xie, S.; Rohrbach, M.; Yan, Z.; Gordo, A.; Feng,
J.; and Kalantidis, Y. 2020b. Decoupling representation and
classifier for long-tailed recognition. In 8th International
Conference on Learning Representations (ICLR).
Kipf, T. N.; and Welling, M. 2016. Semi-supervised classifi-
cation with graph convolutional networks. In Proceedings of
the International Conference on Learning Representations
(ICLR).
Kumar, S.; Hooi, B.; Makhija, D.; Kumar, M.; Faloutsos, C.;
and Subrahmanian, V. 2018. Rev2: Fraudulent user predic-
tion in rating platforms. In Proceedings of the Eleventh ACM
International Conference on Web Search and Data Mining
(WSDM), 333–341.
Li, L.; Zeng, L.; Gao, Z.; Yuan, S.; Bian, Y.; Wu, B.; Zhang,
H.; Lu, C.; Yu, Y.; Liu, W.; et al. 2022. ImDrug: A Bench-
mark for Deep Imbalanced Learning in AI-aided Drug Dis-
covery. arXiv preprint arXiv:2209.07921.
Liu, Z.; Miao, Z.; Zhan, X.; Wang, J.; Gong, B.; and Yu,
S. X. 2019. Large-scale long-tailed recognition in an open
world.
In Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition (CVPR), 2537–
2546.
Liu, Z.; Nguyen, T.-K.; and Fang, Y. 2021.
Tail-GNN:
Tail-Node Graph Neural Networks. In Proceedings of the
27th ACM SIGKDD Conference on Knowledge Discovery
& Data Mining (KDD), 1109–1119.
Ma, H.; Bian, Y.; Rong, Y.; Huang, W.; Xu, T.; Xie, W.; Ye,
G.; and Huang, J. 2020.
Multi-View Graph Neural Net-
works for Molecular Property Prediction.
arXiv preprint
arXiv:2005.13607.
McPherson, M.; Smith-Lovin, L.; and Cook, J. M. 2001.
Birds of a feather: Homophily in social networks. Annual
review of sociology, 27(1): 415–444.
Perozzi, B.; Al-Rfou, R.; and Skiena, S. 2014. Deepwalk:
Online learning of social representations. In Proceedings of
the 20th ACM SIGKDD international conference on Knowl-
edge discovery and data mining (KDD), 701–710.
Shchur, O.; Mumme, M.; Bojchevski, A.; and G¨unnemann,
S. 2018. Pitfalls of graph neural network evaluation. arXiv
preprint arXiv:1811.05868.
Sun, F.-Y.; Hoffmann, J.; Verma, V.; and Tang, J. 2020. Info-
graph: Unsupervised and semi-supervised graph-level rep-
resentation learning via mutual information maximization.
In Proceedings of the International Conference on Learning
Representations (ICLR).
Thakoor, S.; Tallec, C.; Azar, M. G.; Munos, R.; Veliˇckovi´c,
P.; and Valko, M. 2022. Large-Scale Representation Learn-
ing on Graphs via Bootstrapping. In International Confer-
ence on Learning Representations (ICLR).
Van der Maaten, L.; and Hinton, G. 2008. Visualizing data
using t-SNE. Journal of machine learning research, 9(11).
Veliˇckovi´c, P.; Fedus, W.; Hamilton, W. L.; Li`o, P.; Bengio,
Y.; and Hjelm, R. D. 2018. Deep graph infomax. In Pro-
ceedings of the International Conference on Learning Rep-
resentations (ICLR).
Wei, C.; Sohn, K.; Mellina, C.; Yuille, A.; and Yang, F.
2021.
Crest: A class-rebalancing self-training framework
for imbalanced semi-supervised learning. In Proceedings of
the IEEE/CVF Conference on Computer Vision and Pattern
Recognition (CVPR), 10857–10866.
11145


---
## Page 9

Xu, D.; Cheng, W.; Luo, D.; Chen, H.; and Zhang, X.
2021.
InfoGCL: Information-Aware Graph Contrastive
Learning. Advances in Neural Information Processing Sys-
tems (NeurIPS), 34.
Yang, Y.; and Xu, Z. 2020.
Rethinking the value of la-
bels for improving class-imbalanced learning.
In Thirty-
Fourth Advances in Neural Information Processing Systems
(NeurIPS).
You, Y.; Chen, T.; Sui, Y.; Chen, T.; Wang, Z.; and Shen, Y.
2020. Graph contrastive learning with augmentations. In Ad-
vances in neural information processing systems (NeurIPS).
Zhang, H.; Wu, Q.; Yan, J.; Wipf, D.; and Yu, P. S. 2021a.
From canonical correlation analysis to self-supervised graph
neural networks. In Advances in Neural Information Pro-
cessing Systems (NeurIPS).
Zhang, Y.; Kang, B.; Hooi, B.; Yan, S.; and Feng, J.
2021b. Deep long-tailed learning: A survey. arXiv preprint
arXiv:2110.04596.
Zhang, Y.; Wei, X.-S.; Zhou, B.; and Wu, J. 2021c. Bag of
Tricks for Long-Tailed Visual Recognition with Deep Con-
volutional Neural Networks. In Proceedings of the AAAI
Conference on Artificial Intelligence, volume 35, 3447–
3455.
Zhu, Y.; Xu, Y.; Liu, Q.; and Wu, S. 2021a. An Empirical
Study of Graph Contrastive Learning. In Thirty-fifth Con-
ference on Neural Information Processing Systems Datasets
and Benchmarks Track.
Zhu, Y.; Xu, Y.; Yu, F.; Liu, Q.; Wu, S.; and Wang, L. 2021b.
Graph Contrastive Learning with Adaptive Augmentation.
In Proceedings of the Web Conference (WWW), 2069–2080.
11146
