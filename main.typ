#import "@preview/touying:0.4.0": *
#import "@preview/mitex:0.2.2": *
#import "metropolis.typ"
#import "@preview/colorful-boxes:1.3.1": outline-colorbox
#import "@preview/m-jaxon:0.1.1" as m-jaxon
#import "@preview/fletcher:0.4.5" as fletcher: diagram, node, edge
#import "defs.typ": *
#import "@preview/ctheorems:1.1.2": *
#show: thmrules.with(qed-symbol: $square$)

#let lemma = thmbox("lemma", "Lemma", fill: rgb("#eeffee")).with(numbering: none)
#let theorem = thmbox("theorem", "Theorem", fill: rgb("#eeffee")).with(numbering: none)
#let proposition = thmbox("proposition", "Proposition", fill: rgb("#eeffee")).with(numbering: none)
#let corollary = thmplain(
  "corollary",
  "Corollary",
  base: "theorem",
  titlefmt: strong
).with(numbering: none)
#let definition = thmbox("definition", "Definition", inset: (x: 1.2em, top: 1em)).with(numbering: none) 
#let assumption = thmbox("assumption", "Assumption", inset: (x: 1.2em, top: 1em)).with(numbering: none)
#let example = thmplain("example", "Example").with(numbering: none)
#let proof = thmproof("proof", "Proof")

// Themes: default, simple, metropolis, dewdrop, university, aqua
#let s = metropolis.register(aspect-ratio: "16-9")
// #let s = (s.methods.enable-handout-mode)(self: s)//handout-mode
#let s = (s.methods.info)(
  self: s,
  title: [An Alternative Learning Theory with Invariants],
  subtitle: [From An Optimal Recovery Perspective],
  author: [Yang Liu],
  date: datetime.today(),
  institution: [],
)
#let (init, slides, touying-outline, alert) = utils.methods(s)
#show: init

#show strong: alert
#show regex("(e\.g\.)|(i\.e\.)|etc\."): it => text(style: "italic")[#it]
#show regex("\s(ie|eg)\s"): it => {
  let re1 = repr(it).at(2)
  let re2 = repr(it).at(3)
  text(style: "italic")[ #re1.#re2. ]
}

#let (title-slide, slide, focus-slide) = utils.slides(s)
#show: slides.with(title-slide:false, outline-slide:false)


#title-slide(extra:[
  #place(bottom+left,dy: 9em, image("imgs/site-lockup.svg", width: 20em))
])

#touying-outline()
== Dissertation Objective
#outline-colorbox(
  title: "Objective",
  width: auto,
  radius: 2pt,
  centering: false
)[The goal is to establish a learning theory that makes the idea of invariants explicit, and pave the path to an intelligence-driven paradigm of learning.]

#pause

- Focus on the idea of invariants from the ground up
- Focus on the optimal recovery viewpoint

// #pause
// *What is not:*

// - 


== Introduction


- Alternative view of learning theory
  - Recovery maximal learnable space
  - Further shrink down space with invariants
  - Characterization of hypothesis space with generalization
// - Learning theory in Hilbert space/RKHS is well known but limited. 

// - Increasing need for a learning theory targeting Banach space or RKBS.
- Invariants are essential for any learning algorithm: we require more in-depth understanding of invariants.


== Research Questions

*General Theory*

- How do we form hypothesis space based on data?

#pause
- How to best formulate rigorously the concepts of invariants?

    - How does knowledge injection play a role in learning theory?

- What function classes are best for generalization? 

#pause
*Applications*

- How do we understand the domain adaptation using the theory we develop? 
#pause
// - How to apply the learning theory to modern neural networks?

//   - How do understand the generalization behavior better using our theory?

//   - What insights can we bring back from neural networks to general theory?
//     - Is there a hidden convex problem in Reproducing Kernel Banach Space?

// == Alternative View of Learning Theory

// - #emoji.quest Non-iid setting, worst-case error bound

// - Hypothesis space constructed based on data

// - Interpolation learner without overfitting in Banach space (Optimal Recovery)

// - Invariants, knowledge injection as predicates formulations

// - “Self-assurance”/robust in function space
//   - Favorable functions for generalization

// - Alternative approach for understanding neural networks in Banach space
//   - implicit case

// - Connection between convexity of neural networks to RKBS

== Three-Step View of Learning


+ Determine the largest hypothesis space that can be recovered properly with data.
  - Based on compressive sensing, information theory
+ Shrink the initial hypothesis space with invariants
  - Optimal recovery view: given functionals $ell_1, ell_2,..., ell_n$, what set of functions are feasible?
+ Classical data fitting under refined hypothesis space.

#place(right+bottom, dy: -2em)[
  #scale(80%,
  diagram(
    node-stroke: 1pt, node-inset: 10pt,
    edge-stroke: 1pt, debug: 0,
    node((0,0), [1. Initial Hypothesis Recovery], corner-radius: 2pt, extrude: (0, 3)),
    edge("-|>"),
    node((1,0), [2. Learning with Invariants], corner-radius: 2pt, extrude: (0, 3)),
    edge("-|>"),
    node((2,0), [3. Data Fitting], corner-radius: 2pt, extrude: (0, 3))
    // edge("d,r,u,l", "-|>", [Yes], label-pos: 0.1)
  ))
]

== Methodology/ Math Tools


*Overview*

- Formulate a learning theory in Banach space from the approximation theory perspective

- Formulate a unified view of invariants and complete learning problem

- Utilize tools from functional analysis to rigorously derive our learning theory


== Concentration of Measure Theory

Classic works @ledoux2005. 


== Methodology: Approximation Theory

- Very loosely, the aim of approximation theory is to find the optimal function within a given space, that provides the best approximation to the target function. 

- This connects with learning theory closely e.g. @cucker2007. 

  - We aim to find the best approximation in the hypothesis space.


== Methodology: Radon Transform

#mitext(`
\newcommand{\ra}{\mathcal R} 
\newcommand{\s}{\mathbb S} 
\newcommand{\pluto}{\mathbb e} 
$$
\check{f}(p, \xi) = \ra f = \int f(x) \delta(p - \xi \cdot x ) dx
$$

where $(p, \xi) \in \s ^{d-1}\times \R$. Then, for $n\geq 3$ and $n$ is odd, we have

$$
f(x) = \pluto_n \Delta_x^{(n-1)/2} \int_{|\xi|=1}\check{f}(\xi\cdot x, \xi) d\xi
$$
where $\Delta_x$ is the Laplacian operator, and
$$
\pluto_n = \frac{(-1)^{(n-1)/2}}{2(2\pi)^{(n-1)}} = \frac{1}{2}\frac{1}{(2\pi i)^{n-1}}.
$$

== Methodology: Radon Transform

For $n$ is even, we have

$$
f(x) = \frac{\pluto_n}{i\pi}\Delta_x^{(n-2)/2} \int_{|\xi|=1} d\xi \int_{-\infty}^\infty dp \frac{\check{f}_p(p, \xi)}{p-\xi\cdot x}
$$
`)

== Literature Review: Learning Theory

- @vapnik1999 provides an overview of classical learning theory
- @cucker2007 formulate the learning theory from an approximation theory viewpoint

- @vapnik2009 introduces the idea of privileged information
- @vapnik2019@vapnik2019aa introduces the complete learning theory with statistical invariants, i.e. predicates

== Literature Review: Reproducing Kernel Banach Space

- @zhang2009 first proposes the idea

- @lin2022 provides unified definition for RKBS with bilinear form

- @parhi2020 shows that neural networks are representors in RKBS (with semi-norm)
  - implies model specification issue if not with neural networks
    - Q: if neural networks don't provide the best solution, is that because the solution is in difference space? @elbrachter2019

- @bartolucci2023 proposes a revised version RKBS with norm instead of semi-norm, and apply it to neural networks


== Literature Review: Neural Networks

- @wang2020 shows that there's a convex landscape in two-layer networks.

- @yang2019ab@yang2019ac shows that most architectures of neural networks have Gaussian process as the limits

- @parhi2022 shows the optimal solution has the form of neural networks
  - derive a natural occurrence of residual mechanism in the solution

= Learning From An Optimal Recovery Perspective

== Hypothesis Recovery

// - We need first to understand our learning goal. 
- We focus on recoverability, or learnability, of the problem. 

- Since the learning problem is ill-posed, we only consider the hypothesis space that is at least *sparsely* recoverable.

// - Instead of approximation error analysis, we restrict ourselves to the information we have. 

// + Sparse recovery from function space 

// ==== Recovery View on Data

// - Consider we have samples on $X$, then, the learning targets as quantities of interests $Q(x) = y$.
// - The labeled data samples are presented as a problem to recover the quantity of interest.  

// - The learning problems that interest us, however, are finding the recovery operators.
//   - $ cal(F) := { f: (X,Y) -> Q: Q(x) = y, forall (x, y) in (X,Y)} $

== Function Space View

- We bridge the function recovery with binary encoding. Sparsity of the encoded vector represents the complexity of the learning problem. 
  - Only sparse vector can be recovered with "few" samples, analogous to below Nyquist rate.
  - $cal(l)$ bits encoder are denoted as @elbrachter2019
  $ frak(E)^cal(l) := {E: cal(C) arrow {0,1}^cal(l)}, frak(D)^cal(l) := {D:{0,1}^cal(l)-> L^2(Omega)} $

    where $cal(C) in L^2(Omega)$.
  - Minimal coding length:
  $ L(epsilon, cal(C)) := min{cal(l) in NN, exists(E, D) in frak(E)^cal(l)times frak(D)^cal(l): sup_(f in cal(C)) ||D(E(f))-f||_(L^2(Omega)) <= epsilon}$
// - We choose the hypothesis space $cal(H)(epsilon, delta)$ for the a given data set $cal(D)$, such that the recovery error for encoded vector of length $L(epsilon, cal(C)))$ less than $delta$. 

== Encoding Properties

// - We need to consider the $A in RR^(m'*N)$, where $m' = C m$ for some constant. Since single value may not be able to recover the encoded function vector. 
// - We would choose to encode the function subject to distance preserving.
//   - where $A$ is a random matrix, either with i.i.d. Gaussian entries or structured. 
//   - Compatibility conditions:
//     - there exists an $A$, such that $hat(z) = A f$, where $f$ is the encoded function.
//     - $"dist"(hat(z)_i, hat(z)_j) approx "dist"((x_i, y_i), (x_j, y_j))$
//       - Optimizing distance preserving assures the bijective mapping

#assumption[
  Let f be the a function and ${z_i} = {(x_i, y_i)}$ be samples generated from the function $f$ , such that $y_i = f (x_i)$. Then there are encoders $E ∈ frak(E), E′ ∈ frak(E)′$ that can encode both the funtion $f$ and the data samples ${z_i} "as" frak(f) = E( f ), frak(z)_i = E′(z_i)$, such that $ frak(z)_i = A_i frak(f), $
  where $A_i$ is the linear projection.
]



      
// == Encoding Properties

// - The core problem is the entropy of the empirical map
//   - One possible way is to count the possibilities of empirical mapping
//   $ max("dist"(x, x')^d)/min("dist"(x, x')^d) log_2 (c) $
//   - where $m$ is the number of data, and $c$ is number of different outcome.

// == Encoding Length

// - Single digit might not be sufficient to encode the samples. 

// - For instance, MNIST dataset can be nicely visualized in 2d space, hence,  2 + 1 digit would be sufficient for data pair $(x_i, y_i)$.
// - The encoding of the empirical map and the encoding of the data samples are connected by a linear map, ie the measure matrix $A in RR^(m' times N)$.
//   - where $m' = n_c m$, $n_c$ is the length of sample encoding
// - *Challenge:* 
//   - Can the relation between $(x, y)$ and $f$ be represented with $z_(m') = A f_N$? 
//   - i.e. Can we use random matrix $A$ to achieve the same effect as sample from the original function $y = f(x)$?
//   - Or can we avoid talking  about this  exact mapping?

// == Effectiveness of the encoding

// - Since we assume a linear map $A$, and one data pair $(x_i, y_i)$ at least encoded into a $z_m'$ of length $m'$, the length of encoded $f$ can not be too short, or the efficiency of the encoding cannot be too high.
// - Learnability claim requires considerations with properties regarding the measuring matrix.
//   - Connection between matrix $A$ and the sampling process

// Assuming we have above encoding requirements are satisfied, such that each sample carries at least $m'$ bit of information from $z$, then we have a similar PAC learnability proposition as below.

// // #proposition[
// //   For any function that can be binary encoded in length N, $ell$-sparse vector, and each sample data pair can at least provide information equivalent to length $m'$ vector. Then, 
// // ]

// == Error bounds

// - *Challenge:* The recovery error is on encoded function: is there an encoding agnostic error bound?

// - Idea: Consider binary encoding and binary classification case. This seems to corrupt the encoding in a unpredicted way without further assumptions.

//   - Assume that the encoder satisfies some robustness condition: 
//   $ ||f-hat(f)||_2 < C epsilon,  " if" ||x- hat(x)||_1 < epsilon $
//   where $x$ is the encoding of $f$.


// - As problems in compressive sensing, we represent the complexity of functions in terms of bits of coding. 

// == Related to VC dimension (special case)

// - *Same issue as earlier* Can we have encoding agnostic claims?
// - related to growth rate of the encoding length
// - Use VC-dimension as a bridge to connect encoding length and the generalization error
// - Since  $ N <= ss e^(m/(C ss )-1), $
// and if $ phi(italic("VC")) approx C' N $

// - We can bound the generalization error with encoding length $ss, "and" N$.

== Sparsity Recovery

- Any function space can be covered by a closure of hypothesis space $overline(cal(H)_L)$ with large enough length $L$.
- We do not know the ground truth length of the function space

- Given the sample size $m$, we determine the length of the encoded vector of function, so that it can be recovered as $L(epsilon, cal(C))$-sparse vector.
- Since stable recovery require the number of sample 

$ m>= C s ln(e N/s) $
for some constant $C$, hence, $ N <= ss e^(m/(C ss)-1). $
- The largest hypothesis are functions can be encoded with vectors at most length $N$.

== Examples in Hilbert Space
#let hyp = $cal(H)$
- We consider hypothesis space in RKHS as $ B_R := {f in hyp_k, ||f||_K <= R}. $
- Proved in @cucker2007, covering number in Banach space 
$ frak(N)(B_R, eta) <= (2R/eta + 1)^N $
- We also have estimation error bound
$ P_{z in Z^m}(sup_f L_z(f)<= epsilon) >= 1- frak(N)(hyp, epsilon/(8M))exp(-(m epsilon^2)/(8M^4)) $

#proposition[
  Suppose that the hypothesis space is BR as defined in eq. (4.3), such that $| f (x)| ≤ M, ∀ f ∈
B_R$, and suppose that the sample size is $m$. We have $1 − δ$ confidence that $sup_(f in B_R) L_z( f ) ≤ ε$, only if
$
R ≤ ε/(16M) (( δ/∆ )1/N − 1 )
$
where $∆ = exp( − (m ε^2) /(8M^4))$.
]

== Domain Adaptation View of Generalization

- Typical approximation error estimates rely on too much assumptions, since we don't know the ground truth

- Generalization and domain adaptation are inevitably intertwined

- We consider any testing data comes from domain different from training domain 


#proposition[
  Suppose that the empirical distribution of the data is $cal(D)$, the target function is $f ∈ L_2$. We further assume that the range of functions are bounded by $M$. Then, the generalization error on $cal(D)_T$ , for a hypothesis $h$, is $ cal(E)(h) = cal(E)_z( f ) + C M^2 sqrt(8 op("KL")(cal(D)_T || cal(D))) $ for some constant C, where $cal(E)_z$ is the empirical error for the sample z, and KL(·||·) is the KL divergence.
]

// == Research Direction

// - Construct hypothesis testing for validity of the hypothesis space.
//   - Key is to define entropy of empirical map derived from data




// == Existence of encoding and sampling

// Let's consider data $(X, Y)_m$ with empirical mapping that can be encoded with a vector of length $N = n_x + n_y$, such that, there is a matrix $A$, and each data point $(X_i, y_i)$ will be mapped to a distinct value of $v_i$. 

// Such encoding and mapping is feasible, since one can always choose the stacking encoding for $(X, Y)$, and the matrix $A$ would be to pick the corresponding elements. $A$ in this sense, serves as a sampling matrix. 

// The initial learning problem can then be interpreted as recovering the encoded mapping vector sparsely, with a given measure matrix $A in RR^(m times N)$, and measurement vector $V in RR^m$.

// Projection, hypothesis testing

= Invariants Construction
  
== Learning using Statistical Invariants and Predicates

#alert[Suffer from information collapse issue]

- @vapnik2019, @vapnik2019aa Attempt to construct statistical invariants as a way to inject knowledge or intelligence into the learning process

- Minimal empirical success, often less than $1%$ improvements


== LUSI 

- We show that the methods suffer from fundamental issue for the statistical invariants to be effective.
- Predicates
$ cal(P) = 1/m sum_s Phi_s Phi_s^T $

- Loss functions

$ cal(L) = (Y-f(X))^T (gamma I + tau cal(P))(Y-f(X)) $

#lemma[
  Suppose f is a function of a learning task from the data $(X, Y) = {(x_i, y_i)}_i$ using predicates $cal(P)$ and loss function $cal(L)$ constructed earlier. The loss function is equivalent to weighted sum of squares as
  $ cal(L) = sum_i (sum_j (gamma+ lambda_j) q_(j i))(y_i - f(x_i))^2 $
  where $lambda_i$ is the eigenvalue of $tau cal(P)$, and $q_(j i)$ is the entry of orthonormal matrix consisting of the corresponding eigenvectors.
]

== LUSI

#definition("neighborhood density")[
  Let $tilde(D) ⊂ D$ be the training data, then neighborhood error density for a given data pair $(x_i, y_i) ∈ tilde(D)$ defined as
 ̃$ tilde(C)_(x_i) = integral_((x,y)∈D: arg min_(tilde(x)∈ tilde(D)) d(x,tilde(x)) = x_i) C_x d P(x, y). $
 
where d is some metric, L is some loss function, and
$ C_x = (cal(L)(y, f (x)))/(cal(L)(y_i, f (x_i))) $
]

#theorem[
 Suppose that a continuous function (model) f is trained to minimize the loss function $ cal(L) = (Y − f (X))^T (γ I + τ cal(P))(Y − f (X)) " " (4.17) $, where $(X, Y) ∈  tilde(cal(D))$ are training data, sampled from the population $cal(D)$. Assume that the training process can achieve optimality and that the loss for each data point is at least some non-zero small number. Then, the generalization error (calculated using $(X, Y) ∈ cal(D)$)

$ ε = (Y − f (X))^T (Y − f (X)) " " (4.18) $
is smaller, training using eq. (4.17) than training using eq. (4.18) if and only if for all $(x_i, y_i) ∈ cal(D)$,
$
sum_i (| sum_j (gamma+ lambda_i) q_(j i) - tilde(C)_(x_i)| - |1-tilde(C)_(x_i)|)(y_i - f(x_i))^2 < 0 
$
]

== Key issues in LUSI 

- The predicates collapse, by the formulation. Construction information is lost

- Finding predicates using symmetry will not lead to useful predicates, due to theorem earlier

- *We content the key is to utilize and compose information*

== Experiments: Incremental kernel

#place(top+right)[
#image("imgs/pixel.png", width: 50%)
]

- Using incremental kernel, we are able to \ improve the performance, while using \ same information with LUSI cannot. 

  - $K' = K + lambda "CosKernel"(Phi, Phi)$
- Using predicates, LUSI even performs worse.
#v(2cm)
#image("imgs/table1.png", width: 80%)

// == Invariants Formulations: Unified View

// - Invariants are just the partial average of data

// - Any function can be decomposed as an injection, an isomorphism, and a surjection.
//   - The major challenge of a learning problem lies in the learning of the injections.

// - Invariants used in learning methods provide partial information on this injection map.

//   - Haar invariants kernels 
//   - Knowledge injections: known equivalence

== Additive Haar Invariant Kernel

- @haasdonk2004 @haasdonk2007

- With a group of actions $G_0$, we can create invariant kernel as
$ K := integral_(G_0) integral_(G_0) k_0(g x_1, g^prime x_2) d g d g^prime $


#proposition("Additive Invariant Kernel")[
  Let $H$ be normal subgroups of a group of action $G$. Let $cal(A)$ be the set of representatives of the cosets formed by $H$. Then, for an additive kernel $k$, the invariant kernel constructed as below with respect to the Haar measure is additive invariant against actions in $G$:

  $
  K(x_1, x_2) := integral_cal(A) integral_cal(A) k(a x_1, a′ x_2)d a d a′.
  $

The constructed kernel is additive in the sense that the sum of multiple kernels constructed using different
normal groups is also invariant against the actions in G.
]

  
== Optimal Recovery and Learning theory 

- Optimal recovery:

  - Recover a function that satisfies all the functionals constraints.


- Traditional being treated as separated problems
  - Optimal recovery deals with fixed evaluation 
  - Learning theory deals with i.i.d random samples
  - _recent works such as @foucart2022 connect these two_

== Optimal Recovery and Invariants

- Learning can be interpreted as an optimal recovery problem:
  - Invariants are the properties of the target function
  - Data are simply evaluation functionals.

- *Complete learning problem*
  - Data samples ${z_i}$ + functionals ${l_i}$
  - Existing invariants learning algorithms can be interpreted as special cases
  - In LUSI, the @vapnik2019 realize the importance of using functinals, but they didn't provide learning targets for the functionals
  - Bounds can be derived based upon the Chebychev’s ball and classical generalization bounds

// == Predicates and Radon Transform

// - Universality of Radon transform

// - We focus on one types of functionals: Radon transforms. 

//   - Formulate knowledge injection as Radon transform 

//   - Inverse of the Radon transform helps provide useful regularization for hypothesis space 


// == Haar Integration as Radon Transform
// #let int = $integral$

// - Haar integration has been used in invariant construction
//   @haasdonk2005, @mroueh2015

//   $ k(x, x') = int_G_0 int_G_0 k_0(g x, g' x') d g d g' $
//   where $g in G_0$ are group action on $X$.

// - We show that Haar integration is a special case of Radon transform on groups. 
// #let rad = $frak(R)$
// Since $rad_H f(g) = int f(g h) d h$, then, invariants kernel against $G_0 subset G$, is
// $ rad_G_0 rad_G_0 k_0(x, x') = k(x, x') $
// - Is there computational benefits?

// == Radon Transform in RKHS

// How to interpret the Radon transform in Haar kernel induced RKHS?

// such RKHS should be smaller than the space induced by non-invariant kernel. @mroueh2015


== Generalization in Complete Learning Problem
#let rad = $op("rad")$

- Suppose functional invariant operator $cal(I): F arrow RR^n $, with null space
$ cal(N) := {f in F: cal(I) = 0} subset F $
- Further suppose that candidate set $cal(K)$ consists of function can approximate finite dimensional $V subset X$ with error less than $epsilon$. 
- Chebyshev radius of $S subset cal(X)$ defined as
$ op("rad")(S) := inf_cal(X) inf{r: S subset B(a, r)}. $

- According to @devore2017,  we also have $rad(cal(K)) = epsilon mu(cal(N),V)$, where
$ mu(X, Y) = sup_(f in X) (||f||)/"dist"(f, Y). $

// - The wost case radius $R(S) := sup_w rad(S_w)$, where $w$ is the image of $cal(I)$.

#proposition[
  Suppose $cal(K)$ is the candidate space which approximates a finite-dimensional Banach space $V$, which is bounded by $M$, i.e., $cal(K) = { f : "dist"_(L∞) ( f, V) ≤ ε}, cal(I)$ is an invariant operator with null space $cal(N)$ defined earlier. Then we have
$
P_(z in Z^m)(sup_(f in cal(K)) L_z(f) <= xi) >= 1 - ((32 M epsilon mu(cal(N), V))/xi)^N exp(-(-m xi^2)/(8M^4))
$
where $L_z( f ) = integral_(x in z) ( f (x) − y)^2d μ(x, y)$
]



// == Restricted Approximation Error

// // #let asymp = m-jaxon.render("\asymp", inline: true)
// #let asymp = symbol("≍")

// We won't consider the general approximation error, but only the approximation error given a chosen length of encoding.

// Incidentally, this coincides with Kolmogorov $m$-width.

// $ d_m(K, X) := inf{sup_(x in K) inf_(z in X_m) ||x-z||, X_m #[subspace of ] X, dim(X_m) <m} $

// This denotes the error restricting encoding length below a fixed $m$. Since $ d_m(B_2^N, ell_1^N) asymp min {1, sqrt(ln(e N/m)/m)}, $



// == Common Phenomenon on Smallest Eigenvalue

// In optimal recovery, compatibility indicator is given by: @foucart2022aa

// $ mu_(cal(V), op("Id")(Lambda)) = (lambda_min (G_v^(-1/2)C^T G_u^(-1) C G_v^(-1/2)))^(-1/2), $

// and $op("Err")^0_(cal(K), Q) = mu_(cal(V), Q)(Lambda) epsilon$, $cal(K) := {f in F: op("dist")(f, cal(V)) <= epsilon}.$

// While function reconstruction error @smale2005

// $ || tilde(f)-f ||_cal(H) <= (||J||sqrt(sigma^2/delta))/lambda_x^2, $ 

// where $lambda_x = inf_(f in cal(H)) (||S_x f||_(ell^2(x)) )/(||f||_cal(H))$

// == 

// *Data view:*

// _Q: What's the relation between sampling operator/measurement matrix with coveriance matrix of the samples?_

// _Q: generalization error based on random measurement matrix?_

// coherent/RIP (Restricted Isometric Property) in kernel induced sampling operator

// Since the small coherent measurements can lead to better performance, then, such concept can be generalized to kernel matrix. This might lead to dominant eigenvalues, which explains the performance gain via dominant eigenvalue. 

// == Error Bound in Data Space

// - We'd like to talk about error bounds in data space
// - We need to derive the error bound in data space from $ ||f-hat(f)||<epsilon $ to $ P(hat(f)(x) != y). $
// - Further assumptions are required to proceed:
//   - We can simplify the problem by assuming i.i.d or symmetry of the data.
//   - Worst case error

// == Error of Stability

// *Under what conditions, the reconstruction is stable?*

// - We consider the samples as $X_m subset X$, where $dim(X_m) <= m$. Can we construct error bound using Gelfand $m$-width $d^m$ or Kolmogorov $m$-width $d_m$.

// $ d_m(K, X) := inf{sup_(x in K) inf_(z in X_m) ||x-z||, X_m #[subspace of ] X, dim(X_m) <m} $

// - build bounds around the randomness of $X_m$, connecting with $d^m,#[ or ] d_m$.



  

= Characterization of Hypothesis Space
== A Heavy-tailed Function Space


- Motivated by the empirical work @martin2020, heavy tailed empirical spectral density tend to generalize better. 

- This begs the question, is there a certain function class $cal(F) subset cal(B)$, that is more suitable for learning?

- Error bound for stability and robustness #sym.arrow.l.r.stroked sparsity

  - Sparsely recovered hypothesis implies sparsity
  - Stability and recovery error requires heavy tailed eigenvalue

//Missing some other aspects


// == An Special Case: RKHS

// Based on the representor theorem: $f = sum_i a_i phi_(x_i)$. Consider a product space formed by ${a_i}$. 
// Then, the metric entropy of product space is the $product_i op("Ent")(a_i)$. Hence, for minimizing total metric entropy will lead to dominant leading entries. 

// == Local Learnability & Reconstruction

// - Related to local optimal reconstruction
// $ op("lwce")(y, hat(f)) := sup_(f in cal(K)\ Lambda(f) = y) ||f-hat(f)|| $
// - Adapt to situations where sampling is not uniform and function is smooth varying
// - Connect the error with sampling rate:
//   - Higher bandwidth requires higher sample rate #sym.arrow.double error bounds with fixed sample rate



= Applications

== Domain Adaptations: Feasibility and Bounds

#let err = $cal(E)$
- We can also bound the adaptation error with invariants. 

- we define distance between distribution under loss function as

$ 
rho_L (cal(D), cal(D')) &:= sup_(f, f') |err_cal(D)(f, f')-err_(cal(D)')(f, f')| \
&= sup_(f, f')|integral_cal(X) L (d cal(D) - d cal(D'))| \
$
where $L(x) = (f(x)-f'(x))^2$.

==
- Heavily influence by @zhao2019, we can bound adaptation error with the help of invariants.
#proposition[
  Suppose $cal(K)$ is the candidate space which approximates a finite-dimensional Banach space $V$, which is bounded by $M$, i.e., $cal(K) = { f : "dist"_(L∞) (f, V) ≤ ε}, cal(I)$ is an invariant operator with null space $cal(N)$ defined earlier. In addition, suppose that $(cal(D)_S , f_S ), (cal(D)_T , f_T )$ are the source domain and the target domain, respectively.
$ err_T ( f ) ≤ err_S ( f ) + ρ_L (cal(D)_S , cal(D)_T ) + 16ε^2 μ^2(N, V) $
where $ρ_L$, and $μ$ is defined earlier.
]

==



*We will primarily investigate the following two cases:*
/ case 1: There exists functions that are optimal in both domains, but we only learned suboptimal representations from the domain; hence, domain adaptation requires us to update the representation map $phi$. 

/ case 2: We need to adjust the hypothesis class for the target domain, this is typically done by adjusting priors. However, we will focus on exploring the direction through the perspective of predicates.

== Domain Adaptations: Feasibility and Bounds

In *Case 1*, we aim to derive the generalization bound by observing the latent representation. 


- We know feature alignment is not enough, data alignment is also important @zhao2019

- Can we identify the function space that can be adapted well vs. ones that cannot?
  - Representation learning vs. Non-representation learning

- To what extent can we improve this bound using a small amount of data from the target domain?

  - This is a natural extension to @ben-david2006: we utilize the properties of the function spaces 


== Domain Adaptations: Feasibility and Bounds

In *case 2*, we aim to derive error bound for domain adaptation with knowledge injection (predicates). 

- To what extent we can improvement adaptation if there are common functional predicates ${l_i}$.



// == Neural Networks: RKBS


// - Consider the neural networks without layer constraints under RKBS


// == Neural Networks: Convexity

// - There is a hidden convex problem in neural networks

// - Neural networks are the representors in RKBS

// - What are the convexity properties in RKBS?

= Q & A

#text(size: 3em)[#alert[Thanks and Q&A]]

== Appendix

#bibliography("paperpile.bib", style: "chicago-author-date")