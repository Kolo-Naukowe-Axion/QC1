# Transpilation in Qiskit 2.x: Pipeline, SABRE, Optimization Levels, and ODRA-Specific Implications

## Scope and version note

This document is a cleaned and technically tightened research note on the Qiskit transpilation pipeline, with special attention to:

- what `transpile()` actually does,
- what the SABRE family of algorithms does,
- what each `optimization_level` changes in practice,
- what is deterministic versus stochastic,
- how these points matter for experiments on IQM Spark ODRA.

The discussion is aligned with:

- your pinned environment `qiskit==2.1.2`,
- Qiskit 2.x public documentation,
- the SABRE paper,
- the LightSABRE paper.

Important caveat: Qiskit guarantees the public behavior of the built-in stage plugins, but not the exact internal pass ordering of the preset pass managers across minor versions. Therefore, this document distinguishes between:

- public, documented behavior,
- likely implementation details,
- ODRA-specific observations from your own transpilation results.

---

## 1. What `transpile()` does

`transpile()` takes an abstract `QuantumCircuit` and rewrites it into a circuit that can run on a chosen target backend. In Qiskit 2.x, this is done through a staged compiler pipeline built from a preset pass manager.

At a high level, transpilation solves four different problems:

1. **Lowering**: break high-level or unsupported operations into smaller operations.
2. **Placement**: choose which logical qubit should live on which physical qubit.
3. **Routing**: insert extra operations when the hardware connectivity does not directly support a requested two-qubit interaction.
4. **ISA translation and cleanup**: rewrite the result into target-supported instructions and simplify it.

The standard preset pipeline has six named stages:

1. `init`
2. `layout`
3. `routing`
4. `translation`
5. `optimization`
6. `scheduling`

The input is an abstract circuit over virtual qubits.  
The output is a physical circuit over hardware qubits, restricted to the target ISA, optionally with explicit timing.

Internally, Qiskit converts the circuit into a `DAGCircuit` representation and runs a sequence of analysis and transformation passes:

- **Analysis passes** inspect the circuit and write facts into a shared `PropertySet`.
- **Transformation passes** modify the circuit representation.

For most use cases, `transpile()` and `generate_preset_pass_manager()` are the user-facing entry points to the same preset-transpiler machinery.

---

## 2. Constraint precedence and why it matters

When Qiskit decides what hardware constraints to compile against, the precedence is:

1. `target` (highest priority — always wins when provided)
2. loose overrides such as `basis_gates`, `coupling_map`, and `dt` (override corresponding backend fields)
3. `backend` (lowest priority when competing with the above)

The following table from the Qiskit documentation makes this precise:

| User provides | `target` also given | `backend` also given (no `target`) |
| --- | --- | --- |
| `basis_gates` | `target` wins | `basis_gates` wins over backend |
| `coupling_map` | `target` wins | `coupling_map` wins over backend |
| `dt` | `target` wins | `dt` wins over backend |

This matters because experimental claims can easily become misleading if you think you are compiling "for the backend", while in fact you are overriding some of its constraints.

Example:

```python
transpile(circuit, backend=backend, coupling_map=[[0, 1], [1, 2]])
```

This does **not** mean "compile using the backend as-is". It means "compile using the backend, except replace its connectivity with the custom coupling map given here". The explicit `coupling_map` wins over whatever the backend would have provided.

For scientific reporting, you should state explicitly whether you compiled against:

- the backend target exactly as provided,
- a frozen exported target,
- a custom basis or custom coupling map.

---

## 3. The transpilation pipeline in detail

### 3.1 Pipeline overview

The preset transpiler is best understood as a sequence of stages:

```text
abstract circuit
    -> init
    -> layout
    -> routing
    -> translation
    -> optimization
    -> scheduling
    -> runnable hardware circuit
```

Each stage has a distinct job:

- `init`: abstract simplification and unrolling of multi-qubit operations
- `layout`: choose an initial virtual-to-physical qubit mapping
- `routing`: make all required two-qubit interactions hardware-compatible
- `translation`: rewrite all operations into target-supported instructions
- `optimization`: reduce cost on a physical, ISA-level circuit
- `scheduling`: assign explicit timing and delays if requested

Two key scientific lessons follow from this separation:

1. A circuit can gain overhead for two very different reasons:
  - decomposition overhead from non-native gates,
  - routing overhead from non-local connectivity.
2. Only part of transpilation is inherently stochastic:
  - translation is usually deterministic,
  - layout and routing are often heuristic and seed-dependent.

---

### 3.2 `init` stage

#### Purpose

The `init` stage performs high-level work on the still-abstract circuit. It is responsible for:

- lowering operations with more than two qubits,
- handling abstract circuit objects,
- applying high-level logical simplifications before any hardware mapping is chosen.

Its output is still an abstract circuit, but now restricted to one- and two-qubit operations.

#### What it does by optimization level

For the built-in default plugin:

- **Level 0**
  - no abstract optimization,
  - only required unrolling of multi-qubit operations through their `definition`s.
- **Level 1**
  - level 0 behavior,
  - plus simple adjacent inverse cancellation — if two gates that are each other's inverse appear back-to-back on the same qubits (e.g. two consecutive `cx` gates, or an `h` immediately followed by another `h`), they are removed because their combined effect is identity.
- **Levels 2 and 3**
  - level 1 behavior,
  - plus a broader set of abstract optimizations:
    - virtual permutation elision,
    - commutation-aware cancellation,
    - numerical splitting of separable two-qubit operations,
    - removal of negligible operations such as tiny-angle rotations or diagonal gates immediately before measurement.

#### Why this matters

The `init` stage changes the **logical** form of the circuit before hardware details are considered. For research claims, this means:

- a reduction in gate count seen after transpilation may already begin here,
- not all changes later in the pipeline are caused by layout, routing, or native-basis translation.

If your goal is to study pure hardware decomposition of specific gates, `optimization_level=0` is the cleanest starting point because the `init` stage does only the minimum required lowering.

---

### 3.3 `layout` stage

#### Purpose

The layout stage chooses the initial mapping:

```text
virtual qubits -> physical qubits
```

This is the placement problem. It decides where the computation starts on the chip.

The stage also embeds the circuit into the full device width when necessary, including ancilla handling.

#### Why layout is important

A good layout can eliminate routing entirely. A bad layout can force many extra SWAP-like operations later. In practice, layout quality is often one of the largest contributors to hardware overhead.

#### Built-in layout methods

| Method | Algorithm | Summary | Stochastic? |
| --- | --- | --- | --- |
| `trivial` | `TrivialLayout` | Virtual qubit `i` maps to physical qubit `i`. No analysis of the circuit structure. | No |
| `dense` | `DenseLayout` | Finds the densest connected subgraph of the coupling map; prefers high-degree physical qubits. Fast but low quality. | No |
| `sabre` | `SabreLayout` | Runs SABRE routing forwards and backwards to iteratively improve the initial layout. Does **not** attempt VF2 perfect layout. | Yes (seed-dependent) |
| `default` | Composite | Level-dependent: trivial at level 0; VF2 then SABRE fallback at higher levels (see below). | Depends on level |

#### Built-in default behavior by level

- **Level 0**
  - chooses the trivial layout.
- **Level 1**
  - tries trivial layout first,
  - then attempts a perfect layout via `VF2Layout`,
  - then falls back to `SabreLayout` if needed.
- **Level 2**
  - attempts `VF2Layout` first with more effort,
  - then falls back to `SabreLayout`.
- **Level 3**
  - same strategy as level 2, but with the most aggressive layout effort.

#### `VF2Layout`

`VF2Layout` treats layout as a subgraph-isomorphism problem:

- build the interaction graph of the circuit,
- build the connectivity graph of the hardware,
- try to embed the former into the latter without needing routing.

If a perfect embedding exists and is found, routing may become unnecessary — no SWAPs need to be inserted.

This is crucial conceptually:

- if the circuit interaction graph is a subgraph of the hardware graph, zero-routing execution is possible in principle;
- if it is not, routing overhead is unavoidable.

#### Why layout matters for ODRA

ODRA coupling map is:

```python
[[0, 2], [1, 2], [2, 3], [2, 4]]
```

This is a star topology centered on qubit `2`:

```text
  0       1
   \     /
    2 (center)
   / \
  3   4
```

Therefore:

- center-neighbor interactions (0–2, 1–2, 2–3, 2–4) are directly native,
- outer-outer interactions (e.g. 0–1, 0–3, 1–4, 3–4) are **not** directly connected and require routing.

The star has 4 edges. A 5-qubit ring topology (0–1–2–3–4–0) has 5 edges, of which only 2 overlap with the star. This means a ring ansatz is structurally poorly matched to this hardware.

#### Layout and routing interaction

An important architectural subtlety: the Qiskit documentation notes that the layout stage "sometimes subsumes routing." When `SabreLayout` is used, it runs SABRE routing internally as a subroutine for evaluating and refining candidate layouts. In that case, the separate routing stage may find that routing has already been handled and does little or no additional work. This means layout and routing are not always cleanly separable stages — they can overlap in practice.

---

### 3.4 `routing` stage

#### Purpose

The routing stage makes every two-qubit interaction executable on the chosen hardware connectivity graph.

If the circuit wants a two-qubit gate between hardware qubits that are not directly connected, routing inserts extra operations that move or permute logical states so the interaction becomes feasible.

In most workflows, these extra operations are conceptually SWAPs, even if the final target ISA does not contain a literal `swap` gate.

#### Output of routing

Routing updates the effective virtual-to-physical mapping over time. This is why a circuit's initial layout and final layout may differ even if you fixed `initial_layout`.

#### Built-in routing methods

| Method | Algorithm | Summary | Stochastic? | Quality |
| --- | --- | --- | --- | --- |
| `sabre` | `SabreSwap` | Qiskit's enhanced SABRE (LightSABRE). Multi-trial, threaded. | Yes | Best built-in |
| `basic` | `BasicSwap` | Greedy: for each gate in topological order, insert shortest-path SWAPs. | No | Poor |
| `lookahead` | `LookaheadSwap` | Breadth-first search with heuristic pruning. | No | Moderate |
| `stochastic` | `StochasticSwap` | Randomized search for swap mappings. | Yes | Moderate |
| `none` | — | Disables routing entirely. Raises error if routing is needed. | — | — |
| `default` | SABRE-derived | Same as `sabre` in Qiskit 2.x. | Yes | Best built-in |

#### Behavior by optimization level

For built-in defaults, the routing stage uses more effort at higher optimization levels, especially for SABRE-family methods. Public documentation is careful here: the exact pass composition is not frozen, but the broad behavior is:

- **Level 0**
  - minimal effort needed to make the circuit runnable.
- **Level 1**
  - more routing effort than level 0.
- **Level 2**
  - substantially more routing effort and more effort in post-layout improvement.
- **Level 3**
  - the most aggressive built-in routing effort.

After routing, Qiskit also runs `VF2PostLayout` in built-in routing flows to see whether the routed topology can be reassigned onto lower-error physical qubits.

#### Cost of a SWAP on ODRA

A SWAP gate is not a native operation on any current hardware. It must be decomposed into native gates. On ODRA (native basis: `cz` + `r`):

| Operation | Decomposition | Native 2Q gates | Native 1Q gates |
| --- | --- | --- | --- |
| SWAP | 3 CZ + several `r` gates | 3 | ~6 |

This means every routing-inserted SWAP adds **3 additional CZ gates** to the circuit. This is why routing overhead is so expensive: a single SWAP costs as much as 3 native entangling operations, and each of those contributes noise.

#### Why routing is hard

Optimal routing is combinatorial and generally intractable. This is why heuristic methods dominate practical compilers. As a consequence:

- routing is often the main source of transpilation variability,
- different seeds can produce different outputs,
- better routing quality usually costs more compilation time.

#### Scientific interpretation

Routing overhead should be analyzed separately from translation overhead.

For example:

- `CRX` may cost extra gates because it is non-native,
- but it may cost even more if it is placed on non-adjacent qubits and routing must add movement overhead.

These are two different effects and should not be conflated in a hardware-efficiency claim.

---

### 3.5 `translation` stage

#### Purpose

The translation stage rewrites all operations into ones that are supported by the target ISA.

This is where non-native gates are decomposed into native ones.

Examples:

- `cx` may become `cz` plus single-qubit gates,
- `rz` may become several native phased-X-like rotations if the backend does not support a direct `rz`,
- a controlled rotation may become several single-qubit gates plus multiple entangling gates.

#### Built-in translation methods

| Method | Algorithm | Summary | Optimization-level dependent? |
| --- | --- | --- | --- |
| `translator` / `default` | `BasisTranslator` | Symbolic rule-based rewriting using the `SessionEquivalenceLibrary`. Gate-by-gate lookup and decomposition. This is the default and most common method. | No |
| `synthesis` | `UnitarySynthesis` | Collects runs of 1Q/2Q gates into unitary matrices, then resynthesizes from scratch (e.g. KAK decomposition for 2Q unitaries). More expensive but can produce more compact output for simple ISAs. | No |

#### `BasisTranslator`

`BasisTranslator` uses an equivalence library, typically the `SessionEquivalenceLibrary`, to rewrite gates into target-supported operations. For a fixed:

- Qiskit version,
- equivalence library contents,
- target basis,

the symbolic decomposition is deterministic.

This is the central reason why gate-decomposition studies should use level 0 and fixed layouts if the research question is "how expensive is gate X on hardware Y?"

#### What translation does not do

Translation does **not** solve connectivity. If a two-qubit gate is requested on a non-adjacent pair, that is the routing stage's problem.

#### ODRA-specific note on gate names

On IQM Spark ODRA, your backend reports native operations such as:

| IQM native name | Role | Qiskit circuit name |
| --- | --- | --- |
| `prx` | Phased-RX single-qubit rotation: $R(\theta, \varphi) = \exp\bigl(-i\frac{\theta}{2}(\cos\varphi\, X + \sin\varphi\, Y)\bigr)$ | `r` |
| `cz` | Controlled-Z entangling gate | `cz` |
| `cc_prx` | Two-qubit cross-resonance PRX variant | — |
| `prx_12` | PRX on a specific qubit subset | — |
| `measure` | Computational-basis measurement | `measure` |
| `measure_fidelity` | Fidelity-characterization measurement | — |
| `reset_wait` | Qubit reset with wait period | — |

The key mapping for your research: IQM's `prx` gate appears as the Qiskit `r` gate in transpiled circuit output. They are the same operation under different naming conventions. This is why your transpiled circuits show `r` gates rather than `rx`, `ry`, or `rz` — the backend's native single-qubit primitive is PRX, not separate Pauli-axis rotations.

Consequences for decomposition cost:

| Abstract gate | ODRA decomposition | Native `r` gates | Native `cz` gates |
| --- | --- | --- | --- |
| `ry(θ)` | 1 `r` | 1 | 0 |
| `rx(θ)` | 1 `r` | 1 | 0 |
| `rz(θ)` | 3 `r` | 3 | 0 |
| `cz` | 1 `cz` (native) | 0 | 1 |
| `cx` (CNOT) | 4 `r` + 1 `cz` | 4 | 1 |
| `crx(θ)` | ~16 `r` + 2 `cz` | ~16 | 2 |
| `cry(θ)` | ~10 `r` + 2 `cz` | ~10 | 2 |

Note: `crx` and `cry` counts are from your notebook at `optimization_level=0` with adjacent-qubit layout. The `~` prefix reflects that exact single-qubit counts may vary slightly depending on Qiskit version and equivalence-library state.

---

### 3.6 `optimization` stage

#### Purpose

The optimization stage performs low-level improvements on a circuit that is already mapped to hardware qubits and already expressed in target-compatible operations.

This stage is where Qiskit tries to reduce:

- gate count,
- circuit depth,
- two-qubit gate count,
- redundant local structure.

#### Built-in default behavior by level

- **Level 0**
  - empty optimization stage.
- **Level 1**
  - matrix-based resynthesis of runs of one-qubit gates,
  - simple cancellation of consecutive inverse two-qubit gates,
  - repeated until circuit size and depth stabilize.
- **Level 2**
  - level 1 behavior,
  - plus commutation analysis to widen cancellation opportunities,
  - plus a pre-loop matrix-based resynthesis on one- and two-qubit runs.
- **Level 3**
  - level 2 behavior,
  - but with two-qubit matrix-based resynthesis moved into the optimization loop itself,
  - and additional logic to keep the best point when repeated synthesis gives fluctuating outputs.

#### Why this matters

This stage can hide the original decomposition cost of gates if your goal is a clean per-gate study.

For example, if you transpile a circuit containing one controlled rotation at level 3, the final result may reflect:

- decomposition,
- local gate cancellation,
- block resynthesis,
- and retranslation after optimization.

That may be desirable for execution quality, but it is not ideal if your scientific question is "what does this gate translate to on ODRA?" For that question, level 0 is the right baseline.

---

### 3.7 `scheduling` stage

#### Purpose

The scheduling stage makes timing explicit. It inserts `Delay` instructions and can support time-aware transforms such as dynamical decoupling, depending on configuration.

#### Built-in methods

- `default`
  - do nothing unless explicit timing constraints must be respected.
- `alap`
  - as-late-as-possible scheduling.
- `asap`
  - as-soon-as-possible scheduling.

#### Why scheduling is usually irrelevant for gate-decomposition studies

If the research question is circuit structure, native gate counts, or transpilation overhead, scheduling is usually not the main object of study. It becomes important for:

- pulse-aware execution,
- idle-time mitigation,
- wall-clock runtime analysis,
- dynamical decoupling studies.

---

## 4. SABRE in detail

### 4.1 What SABRE is solving

SABRE addresses the qubit-mapping problem for NISQ devices:

- a circuit requests interactions between logical qubits,
- the hardware only allows interactions along edges of a coupling graph,
- we must choose an initial placement and possibly insert SWAPs to realize the circuit.

Both the initial-layout problem and the routing problem are computationally hard. SABRE is a heuristic framework that tries to obtain good solutions quickly.

SABRE stands for:

**SWAP-based BidiREctional heuristic search**

---

### 4.2 Core objects in the original SABRE algorithm

The original SABRE paper works with several key ideas.

#### Current mapping

At every point in the circuit, SABRE tracks a current assignment:

```text
logical qubits -> physical qubits
```

This mapping changes when swaps are inserted.

#### Front layer

The **front layer** is the set of two-qubit gates whose predecessors have already been satisfied. These are the gates that are ready, from the circuit-dependency point of view.

The front layer is important because routing should focus on the interactions that matter immediately, not on the entire remaining circuit equally.

#### Extended set

The **extended set** contains near-future successors of the front-layer gates. This provides limited lookahead, so the router does not make a locally good but globally poor move.

#### Distance matrix

SABRE precomputes distances on the hardware graph. Intuitively, this tells the algorithm how far apart two physical qubits are in terms of routing effort.

---

### 4.3 How SABRE routing works

At a high level, SABRE routing repeats the following loop:

1. Look at the front layer.
2. If a front-layer gate is already executable under the current mapping, execute it logically and advance.
3. If not, generate candidate swaps near the qubits involved in the front layer.
4. Score the candidate swaps heuristically.
5. Choose the best candidate.
6. Update the current mapping and continue.

The key design choice is that SABRE does **not** search all possible swaps over all qubits. It restricts candidates to swaps near the currently relevant part of the circuit. That keeps the search practical.

---

### 4.4 SABRE scoring intuition

The original SABRE heuristic combines three components to score each candidate SWAP:

- a **front-layer term**: average physical distance of the qubit pairs involved in immediately executable gates,
- an **extended-set term**: average physical distance of near-future gate pairs (weighted by a lookahead factor $W$),
- a **decay factor**: multiplicative penalty on qubits that have been recently swapped.

The formula from the paper is:

$$
H_{\text{SWAP}} = \max\bigl(\text{decay}(q_1),\; \text{decay}(q_2)\bigr) \cdot \left[\; \underbrace{\frac{1}{|F|}\sum_{g \in F} D\bigl[\pi(g.q_1)\bigr]\bigl[\pi(g.q_2)\bigr]}_{\text{front-layer cost}} \;+\; W \cdot \underbrace{\frac{1}{|E|}\sum_{g \in E} D\bigl[\pi(g.q_1)\bigr]\bigl[\pi(g.q_2)\bigr]}_{\text{extended-set cost}} \;\right]
$$

Where:

- $F$ is the front layer, $E$ is the extended set,
- $D[i][j]$ is the precomputed shortest-path distance between physical qubits $i$ and $j$,
- $\pi$ is the current logical-to-physical mapping,
- $W \in [0,1)$ controls the weight of the lookahead term,
- $q_1, q_2$ are the two physical qubits involved in the candidate SWAP.

The candidate SWAP with the **lowest** $H$ is chosen.

Key design points:

- **Lower $H$ is better** — it means the front layer becomes easier to execute after this SWAP.
- **Decay is multiplicative, not additive.** This means a recently-swapped qubit inflates the entire score proportionally, discouraging repeated use of the same qubits and thus encouraging parallelism.
- **The lookahead term prevents myopia.** Without it, SABRE might choose a SWAP that helps the current front layer but makes future gates harder.
- By tuning the decay parameter $\delta$, one trades off gate count against circuit depth.

---

### 4.5 Bidirectional idea

The most distinctive idea in SABRE is bidirectional refinement.

Routing quality depends strongly on the initial layout. But how do we pick a good initial layout before routing has begun?

SABRE's answer is:

1. start with a candidate initial mapping (possibly random),
2. route the circuit in forward gate-dependency order — this produces a final mapping,
3. reverse the gate-dependency order of the circuit DAG,
4. use the final mapping from step 2 as the initial mapping for routing the reversed sequence,
5. the resulting mapping from step 4 becomes the new candidate initial mapping,
6. repeat steps 2–5 for several iterations.

Note: "reversing the circuit" here means reversing the topological order in which gates are processed, **not** taking the adjoint/inverse of each gate. For routing purposes, only the qubit-pair interactions matter, not the gate types, so the reversal is purely about processing order.

This lets information from later parts of the circuit influence the starting placement. In other words, the initial layout is improved by looking at the whole circuit, not just the first few gates.

In Qiskit, this idea appears in `SabreLayout`, where routing is used internally as a subroutine to evaluate and refine candidate layouts. `SabreLayout` is not the same as `SabreSwap`:

| Component | Role | Used in |
| --- | --- | --- |
| `SabreLayout` | Chooses the **initial mapping** by running routing forwards and backwards to refine placement | Layout stage |
| `SabreSwap` | Inserts **SWAP operations** to make a given circuit executable under a fixed initial mapping | Routing stage |

`SabreLayout` uses `SabreSwap` internally as a subroutine — it routes the circuit multiple times to judge which starting placement produces the least overhead, then selects the best one.

---

### 4.6 Why SABRE became dominant

SABRE became influential because it offers a good balance:

- much better quality than simple greedy routing,
- much cheaper than exhaustive search,
- effective on realistic NISQ-scale circuits.

For quantum machine learning circuits, this matters a lot because ansatze can be:

- deep,
- highly repetitive,
- transpiled many times across seeds, folds, and experiments.

Routing quality directly affects both:

- final physical depth,
- accumulated two-qubit error.

---

## 5. LightSABRE and Qiskit's modern SABRE-based behavior

Qiskit's modern built-in SABRE family is based on the LightSABRE work.

### 5.1 Main ideas in LightSABRE

The LightSABRE paper improves the practical implementation of SABRE in several ways:

- **relative scoring**
  - evaluate the change in heuristic cost more cheaply rather than recomputing everything from scratch;
- **multi-trial search**
  - run several candidate searches with different seeds or initializations and keep the best result;
- **better initial seeding**
  - include stronger layout candidates rather than relying purely on random starts;
- **release-valve logic**
  - escape local stalls when the heuristic is not making progress;
- **faster implementation**
  - substantial implementation-level speedups;
- **better practical scaling**
  - improved performance on larger circuits and realistic workloads.

### 5.2 SABRE trial configuration by optimization level

The following table summarizes the SABRE-family parameters as described in the LightSABRE paper. These are representative of the Qiskit default preset behavior, though exact values are implementation-dependent and may evolve between minor releases.

| Parameter | Level 0 | Level 1 | Level 2 | Level 3 |
| --- | --- | --- | --- | --- |
| Layout trials | 1 | 5 | 20 | 20 |
| Swap (routing) trials | 1 | 5 | 20 | 20 |
| Forward-backward iterations | 1 | 3 | 4 | 4 |
| Heuristic variant | basic (no decay) | decay | decay | decay |

Key observations:

- **Level 0 uses only 1 trial with the basic heuristic** — no decay, no multi-trial diversity. This is why level 0 routing is fast but can be poor.
- **Levels 2 and 3 have the same SABRE parameters.** The difference between them lies in the optimization stage, not in layout/routing trial counts.
- **The decay heuristic** at levels 1+ enables the depth-vs-gate-count trade-off discussed in §4.4.

### 5.3 What Qiskit publicly guarantees versus what can change

Publicly, you can rely on the following broad statements:

- `routing_method="sabre"` uses a SABRE-derived routing algorithm.
- `layout_method="sabre"` uses SABRE-derived bidirectional layout refinement.
- higher optimization levels generally devote more effort to layout/routing quality.
- seeded built-in passes are deterministic on a fixed machine and fixed environment.

What Qiskit does **not** promise as stable across minor versions:

- exact pass ordering,
- exact number of trials,
- exact internal heuristics,
- exact tie-breaking behavior beyond seeded determinism guarantees.

For a research document, this means:

- explain SABRE conceptually and correctly,
- avoid hard-coding unstable implementation details unless you directly inspect the exact versioned pass manager,
- pin the software stack when claiming numerical results.

---

## 6. Optimization levels: what each one specifically does

This section summarizes the default preset behavior at each `optimization_level` in Qiskit 2.x.

Important note: the table below is meant as a faithful summary of the **documented broad behavior** of the built-in preset pass managers. The exact internal pass sequence may evolve between minor releases.

### 6.1 Compact comparison table


| Stage          | Level 0                              | Level 1                                            | Level 2                                                           | Level 3                                          |
| -------------- | ------------------------------------ | -------------------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------ |
| `init`         | required unrolling only              | level 0 + simple inverse cancellation              | level 1 + broader abstract optimizations                          | same broad class as level 2                      |
| `layout`       | trivial layout                       | trivial, then try perfect VF2, then SABRE fallback | stronger VF2 search, then SABRE fallback                          | strongest built-in layout effort                 |
| `routing`      | minimal routing effort needed to run | more effort than level 0                           | stronger routing effort                                           | strongest routing effort                         |
| `translation`  | basis translation to target ISA      | same role                                          | same role                                                         | same role                                        |
| `optimization` | empty                                | 1Q resynthesis + simple 2Q inverse cancellation    | level 1 + commutation-aware widening + pre-loop 1Q/2Q resynthesis | level 2 + more aggressive in-loop 2Q resynthesis |
| `scheduling`   | none unless requested                | none unless requested                              | none unless requested                                             | none unless requested                            |


### 6.2 Level 0

**Interpretation:** "Make it runnable, but do not spend effort improving it."

Specifically:

- `init` does only the lowering that is required.
- `layout` uses the trivial mapping by default.
- `routing` is only as elaborate as needed to satisfy connectivity.
- `translation` rewrites gates into the target ISA.
- `optimization` is empty.
- `scheduling` is inactive unless explicitly requested.

Best use cases:

- decomposition studies,
- fair baseline comparisons,
- debugging where you want to see the raw compiler cost,
- experiments where you want to minimize hidden optimization effects.

Main limitation:

- results can be structurally far from the best executable circuit.

### 6.3 Level 1

**Interpretation:** "Apply light cleanup with modest extra compile effort."

Specifically:

- `init` adds simple inverse-gate cancellation.
- `layout` first tries trivial placement, then a perfect embedding via VF2, then SABRE fallback.
- `routing` uses more effort than level 0.
- `optimization` starts simplifying one-qubit runs and canceling obviously redundant neighboring two-qubit structure.

Best use cases:

- fast compilation with some cleanup,
- moderate-size sweeps where compilation time matters.

Main trade-off:

- more compiler work than level 0,
- still not the strongest optimization level for deep difficult circuits.

### 6.4 Level 2

**Interpretation:** "Default practical balance between compile time and quality."

This is also the documented default when `optimization_level=None`.

Specifically:

- `init` enables broader logical simplifications such as commutation-aware cleanup and removal of negligible operations.
- `layout` gives more serious effort to finding a perfect VF2 embedding before falling back to SABRE.
- `routing` receives more effort than level 1.
- `optimization` performs stronger local simplification, including pre-loop resynthesis of one- and two-qubit runs.

Best use cases:

- general benchmarking,
- production-like compilation when you want a strong default,
- studies where you want realistic "normal user" transpilation settings.

Main trade-off:

- less interpretable than level 0 for raw decomposition analysis,
- more compile-time cost.

### 6.5 Level 3

**Interpretation:** "Spend the most effort trying to improve the circuit."

Specifically:

- uses the strongest built-in layout and routing effort,
- keeps the strongest abstract simplifications from level 2,
- applies more aggressive two-qubit resynthesis inside the optimization loop,
- may run much longer on large circuits.

Best use cases:

- final performance-oriented compilation,
- studies asking whether an ansatz advantage survives aggressive optimization.

Main trade-offs:

- highest compile time,
- output can be harder to interpret mechanistically,
- not guaranteed to be best for every metric on every backend.

### 6.6 Practical recommendation for research

Use the optimization levels for different purposes:

- **Level 0** for decomposition-only and mechanism-focused analysis.
- **Level 1** for a light-optimization baseline.
- **Level 2** for the default practical compilation baseline.
- **Level 3** to test whether conclusions survive aggressive compiler help.

Do not treat these levels as simply "same pipeline but more optimization". The chosen stage behavior changes materially between levels, especially in `layout`, `routing`, and `optimization`.

---

## 7. What is predictable and what is not

### 7.1 Summary table

| Transpilation aspect | Deterministic? | Depends on | How to control |
| --- | --- | --- | --- |
| Gate decomposition (`BasisTranslator`) | **Yes** | Qiskit version, equivalence library, target basis | Pin Qiskit version, use level 0 |
| Layout choice (level 0, trivial) | **Yes** | — | Use `layout_method="trivial"` |
| Layout choice (level 1+, VF2) | **Yes** if perfect layout exists | Circuit structure, coupling map | — |
| Layout choice (SABRE-based) | **No** — stochastic | Seed, number of trials, heuristic | Fix `seed_transpiler` |
| Routing (SABRE) | **No** — stochastic | Seed, layout, circuit structure | Fix `seed_transpiler` |
| Optimization passes | **Yes** given fixed input | Input circuit after routing | Pin Qiskit version |
| Scheduling | **Yes** | Backend timing data | — |
| **End-to-end transpile output** | **No** in general | All of the above | Pin everything + fix seed |

### 7.2 Predictable parts

For fixed:

- Qiskit version,
- equivalence library,
- target,
- optimization settings,
- random seed where relevant,

the following are either deterministic or close to deterministic in the intended scientific sense:

- symbolic translation of non-native gates by `BasisTranslator`,
- level-0 decomposition patterns under a fixed target,
- seeded outputs of built-in stochastic passes in a frozen environment.

### 7.3 Not fully predictable parts

The main nontrivial variability comes from layout and routing heuristics.

Even with fixed seed, outputs can differ across:

- Qiskit versions,
- operating systems (different math-library implementations),
- different underlying package versions,
- different CPU instruction sets (e.g. fused multiply-add rounding differences).

Qiskit guarantees that for a fixed seed on a fixed machine with a fixed environment, built-in passes produce deterministic output regardless of thread count (unless `QISKIT_SABRE_ALL_THREADS` is explicitly set).

So the correct scientific position is:

- gate decomposition is usually analytically predictable,
- routing overhead is typically only statistically characterizable unless you fix the full environment and seed.

### 7.4 Reproducibility checklist for publications

| Item | Action |
| --- | --- |
| Qiskit version | Pin in `requirements.txt` (you use `qiskit==2.1.2`) |
| Transpiler seed | Set `seed_transpiler` and report it |
| Backend/target | Freeze and export the `Target` object, or document the calibration date |
| Optimization level | Report explicitly |
| Layout/routing method | Report if non-default |
| OS and Python version | Document in supplementary materials |
| Circuit hash | Store transpiled QASM or circuit hash per run |

---

## 8. ODRA-specific implications for your project

### 8.1 Native-gate interpretation

For your ODRA backend, the important native picture is:

- entangling operation: `cz`
- one-qubit native family: PRX/phased-X-like operations exposed through IQM's target

Therefore:

- `cz` is genuinely native,
- `cx` is not native and must be synthesized,
- `crx` and `cry` are not native and must be synthesized,
- `rz` is not directly native in the same way as `cz` and must be built from native one-qubit rotations.

### 8.2 Topology interpretation

ODRA's star coupling map `[[0,2],[1,2],[2,3],[2,4]]` has 4 edges. The following table shows how common ansatz interaction patterns map onto it:

| Interaction pattern | Edges needed | Edges present in star | Edges missing | Routing required? |
| --- | --- | --- | --- | --- |
| Star (all through center) | 4: (0,2),(1,2),(2,3),(2,4) | 4/4 | 0 | **No** |
| Linear chain 0–1–2–3–4 | 4: (0,1),(1,2),(2,3),(3,4) | 2/4 | (0,1),(3,4) | Yes |
| Ring 0–1–2–3–4–0 | 5: (0,1),(1,2),(2,3),(3,4),(4,0) | 2/5 | (0,1),(3,4),(4,0) | Yes |
| All-to-all on 5 qubits | 10 | 4/10 | 6 | Yes (heavy) |
| Nearest-neighbor to center only | varies | all present by construction | 0 | **No** |

This means:

- ansatze based on star-like connectivity are structurally well-aligned,
- ring-like ansatze will incur routing overhead on most of their edges,
- the choice of entanglement topology in the ansatz design has a direct, measurable impact on transpiled circuit cost.

### 8.3 Implication for your notebook results

Your notebook results are consistent with this interpretation:

- `cz -> cz`
- `cx -> 4 r + 1 cz`
- `crx -> 2 cz + many r`
- `cry -> 2 cz + many r`
- `rz -> 3 r`

This is exactly the kind of evidence that should be used in the project report. It shows:

1. decomposition overhead from non-native gates,
2. the advantage of native `cz`,
3. the non-native cost of `rz`,
4. why decomposition-only studies must control layout to avoid mixing in routing overhead.

---

## 9. Recommended methodology for your hardware-efficiency study

If your research goal is to argue that one ansatz is more hardware-efficient than another on ODRA, a rigorous protocol is:

### Step 1: isolate decomposition cost

Compile with:

```python
transpile(
    circuit,
    backend=backend,
    optimization_level=0,
    routing_method="none",
    initial_layout=...,
)
```

when possible. If routing is required, that is itself evidence that the interaction graph is not directly embeddable.

### Step 2: measure routing cost separately

Run multiple transpiles with:

- optimization levels `0, 1, 2, 3`,
- several `seed_transpiler` values,
- optionally alternative layout and routing methods.

Track:

- total gate count,
- two-qubit gate count,
- depth,
- routed SWAP-equivalent overhead,
- transpilation time.

### Step 3: relate structure to hardware results

On hardware or a calibrated noise model, compare:

- inference quality,
- native two-qubit counts,
- total depth,
- stability across transpiler seeds.

### Step 4: report deterministic and stochastic parts separately

Your argument should explicitly separate:

- **deterministic structural cost**:
  - non-native gate decomposition,
- **stochastic structural cost**:
  - layout/routing overhead,
- **hardware/noise cost**:
  - fidelity loss at execution time.

That separation makes the claim much stronger scientifically.

---

## 10. Key takeaways

1. `transpile()` is not just basis translation. It is a full staged compiler pipeline.
2. The biggest conceptual distinction is between:
  - decomposition overhead,
  - routing overhead.
3. `BasisTranslator`-based decomposition is usually deterministic for a fixed environment.
4. Layout and routing are the main sources of stochastic transpilation variability.
5. SABRE is a heuristic method for choosing layouts and inserting swaps using front-layer and lookahead information.
6. Higher optimization levels do not merely "optimize more"; they change how several stages behave.
7. For ODRA, `cz` is native, while `cx`, `crx`, `cry`, and `rz` are not.
8. ODRA's topology is star-like, so center-neighbor interactions are naturally favored over ring-like interaction graphs.
9. For gate-decomposition analysis, `optimization_level=0` is the correct baseline.
10. For realistic performance studies, level 2 is the most natural default comparison point, with level 3 used as an aggressive-compilation stress test.

---

## References

1. G. Li, Y. Ding, Y. Xie, *Tackling the Qubit Mapping Problem for NISQ-Era Quantum Devices*, arXiv:1809.02573, 2019.
2. H. Zou, M. Treinish, K. Hartman, A. Ivrii, J. Lishman, *LightSABRE: A Lightweight and Enhanced SABRE Algorithm*, arXiv:2409.08368, 2024.
3. IBM Quantum Documentation, *Qiskit Transpiler Overview*, Qiskit 2.x documentation.
4. IBM Quantum Documentation, *`qiskit.compiler.transpile` API reference*, Qiskit 2.x documentation.
5. IBM Quantum Documentation, *Preset pass managers and transpiler stages*, Qiskit 2.x documentation.

