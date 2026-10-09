# FHETCH Polynomial IR Integration into the HEIR Compiler

## Full Scoping & Implementation Plan

---

## Executive Summary

This plan outlines the full scoping of integrating the FHETCH Polynomial IR as a standardized hardware abstraction layer backend into HEIR. FHETCH Polynomial IR is a **hardware-level IR** (category 3 in HEIR's backend taxonomy), lowering from HEIR's `polynomial` dialect with `rns`-of-`mod_arith`-typed coefficients.

**Key objectives:**
1. Define FHETCH Polynomial IR as an HEIR exit dialect, here called the `fhetch` dialect
2. Establish the lowering from `polynomial` to `fhetch`, fed by HEIR's scheme-to-`polynomial` passes
3. Create code generation that emits `fhetch`
4. Define backend-specific passes for RNS/NTT optimization and resource allocation
5. Build test infrastructure and examples

> **Status note (audited 2026-08-24 against `main` @ `ae757edc0`).** This plan was
> drafted before auditing HEIR. Substantial parts of Phases 2 and 4 already exist
> upstream and should be reused rather than rebuilt, and Phase 3's lowering chain
> as originally drafted was wrong. Sections below have been revised; see
> [Phase 0](#phase-0-audit-of-existing-heir-capabilities) for the summary of what
> changed and why.
>
> **Implementation status (2026-10-14, branch `fhetch`).** Phase 2 (the `fhetch`
> dialect, its verifiers and folders) and Phase 3's lowering
> (`--polynomial-to-fhetch`) are implemented and tested. Implementing them
> settled a question the draft left open: FHETCH instructions are
> *single-residue*, so `fhetch` ops take polynomials over one `!mod_arith.int<q>`
> and the lowering splits RNS values into one value per limb. Sections 2.1, 2.2,
> 3.2, 3.3, 4.2, 5.2, 6.1, 8.3 and Appendix B have been revised to match the code.

---

## Phase 0: Audit of Existing HEIR Capabilities

### 0.1 HEIR already documents FHETCH as the category-3 example

`docs/content/en/docs/Guides/new_backend.md` names <https://fhetch.org/> as its
example of a category-3 backend, and specifies that such an exit dialect
"would have its lowering start from the `polynomial` dialect, with
`rns`-of-`mod_arith`-typed coefficients". It adds the caveat this plan must
account for:

> the relevant FHE cryptosystem must be implemented as lowering passes in HEIR.
> As of 2026, this is in various states of progress (Cf.
> [#2886](https://github.com/google/heir/issues/2886) for CKKS).

### 0.2 The blocking gap: scheme ops do not reach `polynomial`

Measured empirically on a CKKS dot product:

```bash
heir-opt --mlir-to-ckks dot.mlir              # 4 ckks.rotate, 3 ckks.rescale, 1 ckks.relinearize
heir-opt --ckks-decompose-relinearize --ckks-to-lwe   # all 8 survive unconverted
```

`lib/Dialect/CKKS/Transforms/CKKSToLWE.cpp` is 40 lines covering exactly seven
elementwise ops (`add`, `add_plain`, `sub`, `sub_plain`, `negate`, `mul`,
`mul_plain`). `ckks.rotate`, `ckks.rescale`, `ckks.level_reduce`, and
`ckks.bootstrap` have **no** lowering to `lwe`/`polynomial`.
`ckks-decompose-relinearize` bails at `if (!op.getKeySwitchingKey())` because
`--mlir-to-ckks` does not materialize key operands.

The only consumers of `ckks::RotateOp`/`RelinearizeOp`/`RescaleOp` are the
category-1 library backends (`openfhe`, `lattigo`, `jaxiteword`). **Lowering
`ckks` directly to `fhetch`, as originally drafted in Phase 3, would make
`fhetch` a category-1 dialect wearing a category-3 label** — forfeiting the
point of a shared hardware abstraction layer. Phase 3 has been rewritten
accordingly.

### 0.3 Machinery to reuse, not rebuild

| Originally proposed here | Already in HEIR |
|---|---|
| `#fhetch.ntt_domain_attr` | `!polynomial.polynomial<ring=…, form=coeff\|eval>` — the NTT domain is carried *in the type* (`PolynomialEnums.td`) |
| §4.2.1 NTT/INTT hoisting pass | `--convert-polynomial-mul-to-ntt` ("Optimally inserts NTT/INTT into a Polynomial IR") |
| INTT∘NTT elimination | `INTTAfterNTT` / `NTTAfterINTT` patterns in `PolynomialCanonicalization.td` |
| §3.2 Step 1 `RotationAnalysis` | `--rotation-analysis` |
| `fhetch.polynomial<degree, rns_limbs, limb_width>` | all three derivable from `#polynomial.ring<coefficientType = !rns.rns<!mod_arith.int<…>>, polynomialModulus = …>` |
| Key-switch / mod-down decomposition | `--lwe-decompose` (`DecomposeKeySwitchPattern`, `DecomposeModDownPattern`) |
| NTT twiddle-factor setup | `--attach-ntt-roots` |

### 0.4 Prerequisite bug

`--lwe-to-polynomial` is broken for all three ciphertext-plaintext ops:

- `lwe.rmul_plain` **crashes** on an assertion (`cast<RankedTensorType>` of a
  non-tensor) at `lib/Dialect/LWE/Conversions/LWEToPolynomial/LWEToPolynomial.cpp:510`
- `lwe.radd_plain` / `lwe.rsub_plain` emit invalid IR
  (`'polynomial.add' op requires the same type for all operands and results`)

Root cause is one type-converter mismatch: `LWEPlaintextType` converts to a bare
`!polynomial.polynomial` while ciphertexts convert to `tensor<2x!poly>`;
`ConvertRMulPlain` assumes the plaintext is `tensor<1x!poly>`. CI is green
because `tests/Dialect/LWE/Conversions/lwe_to_polynomial/` has no test for any
of the three. Every realistic CKKS program hits `rmul_plain`, so this is on the
critical path.

**Deliverables:**
- [ ] Fix the `LWEToPolynomial` plaintext type mismatch; add the three missing lit tests
      — a separate track. It gates only the scheme-to-`polynomial` stretch, so
      Phases 3–5 proceed from hand-written `polynomial` IR in parallel
- [ ] File/track the CKKS-to-`polynomial` gap against upstream #2886

---

## Phase 1: Requirements & Specification Alignment

### 1.1 FHETCH Polynomial IR Specification Analysis

**Inputs needed:**
- [x] FHETCH Polynomial IR Draft Specification (https://fhetch.org/)
- [x] Opcode definitions for ring arithmetic: ADD, SUB, MUL, ADDI, SUBI, MULI,
      NTT1/NTT2, INTT1/INTT2, MORPH1/MORPH2 (§2.1)
- [ ] Operand encoding (polynomial registers, immediate moduli, batch sizes)
- [ ] Memory/storage model (polynomial storage, coefficient layout)
- [x] Type system: one residue per instruction; degree from the ring's
      polynomial modulus, modulus and width from `!mod_arith.int<q : iW>` (§2.1)
- [ ] Metadata & control structures (key generation, rotation indexes, etc.)

**Deliverables:**
- [x] Opcode-to-operation mapping: polynomial IR operations to FHETCH opcodes
      (§2.1 table, §3.2 Step 3)
- [x] Type mapping matrix: HEIR types to FHETCH types (§2.1, §3.2 Step 2)

### 1.2 HEIR Polynomial Dialect Assessment

**Current state analysis:**
- [x] Review `lib/Dialect/Polynomial/IR/`: univariate polynomial types and operations
- [x] Review `lib/Dialect/RNS/`: RNS form representation
- [x] Review `lib/Dialect/ModArith/`: modular arithmetic operations
- [x] Audit current lowering paths: scheme (CKKS/BGV) to polynomial to RNS
- [x] Identify gaps: NTT domain operations, hardware-specific annotations, semantics

**Findings.**

*Types.* `!polynomial.polynomial<ring=…, form=…>` carries everything the drafted
`fhetch.polynomial` wanted:

```mlir
!poly = !polynomial.polynomial<
  ring = <coefficientType = !rns.rns<!mod_arith.int<35184372744193 : i64>, …>,
          polynomialModulus = <1 + x**1024>>,
  form = coeff>          // or form = eval  (i.e. NTT domain)
```

- polynomial degree ⇐ `polynomialModulus` (`x**N + 1`)
- RNS limb count ⇐ arity of the `!rns.rns<…>` coefficient type
- limb width ⇐ storage type of each `!mod_arith.int<q : iW>`
- NTT domain ⇐ the `form` parameter (`Polynomial_FormEnum`: `COEFF` | `EVAL`)

`RingAttr` itself is only `(coefficientType, optional polynomialModulus)` —
everything else is derived.

*Ops.* `polynomial` has `add`, `sub`, `mul`, `mul_scalar`, `ntt`, `intt`,
`mod_switch`, `extract_slice`, `monic_monomial_mul`, `apply_coefficientwise`.
`mod_arith` has `add`/`sub`/`mul`/`mac`/`barrett_reduce`/`subifge`/`mod_switch`.
Between them these cover the arithmetic half of the drafted `fhetch` opcode
table; the traits `SameOperandsAndResultForm` and `FixedFormCoeff` already
enforce the NTT-domain consistency rules drafted in §2.1.

*Lowering paths.* `ckks-to-lwe` → `lwe-decompose` → `lwe-to-polynomial` exists as
a chain, but is incomplete at both ends — see [Phase 0](#phase-0-audit-of-existing-heir-capabilities)
§0.2 (missing rotate/rescale) and §0.4 (plaintext-op bug).

**Deliverables:**
- [x] HEIR polynomial IR summary: what can be reused, what needs to be extended → §0.3
- [x] Gap analysis report: how completely from polynomial IR to FHETCH → §0.2
- [ ] Design proposal for CKKS `rotate`/`rescale` → `lwe`/`polynomial` lowering (upstream #2886)

---

## Phase 2: Exit Dialect Design (`fhetch` Dialect)

### 2.1 Define FHETCH Exit Dialect

**Scope:** Mirror FHETCH Polynomial IR specification as an MLIR dialect.

**Structure** (following HEIR examples like `lattigo` and `openfhe`):

HEIR builds with Bazel, and its build files are named `BUILD` (not `BUILD.bazel`,
and never `CMakeLists.txt`). Tests live under `tests/`, mirroring the `lib/` path
— not in a `test/` subdirectory of the dialect.

```
lib/Dialect/Fhetch/
├── IR/
│   ├── BUILD
│   ├── FhetchDialect.h / .cpp / .td
│   ├── FhetchOps.h / .cpp / .td       # Operation definitions
│   └── FhetchTypes.h / .cpp / .td     # Type definitions (deferred, see below)
└── Transforms/
    ├── BUILD
    └── [optimization passes]

lib/Dialect/Polynomial/Conversions/
└── PolynomialToFhetch/                 # the lowering (§3), next to PolynomialToModArith

tests/Dialect/Fhetch/IR/                # ops, verifier and folding tests
tests/Dialect/Polynomial/Conversions/polynomial_to_fhetch/
```

Use `scripts/templates/templates.py` to generate the boilerplate (it needs
`python-fire` and `jinja2`); registration in `heir-opt.cpp` and the build deps
are still manual.

**Key types to define.**

> **Revised.** The draft below duplicated `!polynomial.polynomial`. Degree, RNS
> limb count, limb width and NTT domain are *already* carried by the HEIR type
> (§1.2), and a parallel `fhetch.polynomial` would force a lossy round-trip and
> defeat the `polynomial` canonicalizations and passes listed in §0.3.
>
> ```
> // NOT THIS:
> fhetch.polynomial<degree: i64, rns_limbs: i64, limb_width: i64>
> fhetch.ntt_domain_attr
> fhetch.modulus<value: i64>
> ```
>
> Instead, `fhetch` ops should take `!polynomial.polynomial` operands directly,
> and define new types **only** for concepts `polynomial` genuinely lacks.
>
> **Revised again (implemented).** The operands are constrained to a *single*
> `!mod_arith.int<q>` coefficient type, not RNS-of-`mod_arith`: every FHETCH
> instruction carries exactly one modulus, so RNS is expanded limb by limb by the
> lowering (§3.2). The dialect defines no types yet; the two below are deferred
> until resource allocation and key switching need them:

```
// Hardware polynomial register / storage slot — no HEIR equivalent.
// Introduced by the resource-allocation pass (§4.2.3), not by lowering.
!fhetch.poly_ref<...>

// Key material as a typed SSA value, if the accelerator takes it as an operand
// (cf. OpenFHE's CryptoContext). Check against `key_mgmt` dialect first.
!fhetch.eval_key<...>
```

Immediate moduli come from the operand's `!mod_arith.int<q : iW>` type; rotation
indices are plain `IndexAttr`/`I64Attr` operands as in `ckks.rotate`. Reserve new
attributes for hardware facts HEIR cannot express — batch size, latency,
throughput, target name.

**Key operations to define** (implemented in `FhetchOps.td`):

All operands are single-residue polynomials, `!polynomial.polynomial` over one
`!mod_arith.int<q>` with a polynomial modulus; "coeff"/"eval" is the type's
`form`.

| Operation | ISA opcode | FHETCH call | Operands | Result | Lowered from |
|-----------|------------|-------------|----------|--------|--------------|
| `fhetch.add` | ADD | `sr_addp` | 2 × poly, same form | poly | `polynomial.add` |
| `fhetch.sub` | SUB | `sr_subp` (also `sr_negp` as `0 - a`) | 2 × poly, same form | poly | `polynomial.sub` |
| `fhetch.mul` | MUL | `sr_mulp` | 2 × poly, same form | poly | `polynomial.mul`, eval form only |
| `fhetch.addi` | ADDI | `sr_addps` / `sr_addps_coeff` | poly, scalar | poly | — |
| `fhetch.subi` | SUBI | `sr_subps` / `sr_subps_coeff` | poly, scalar | poly | — |
| `fhetch.muli` | MULI | `sr_mulps` | poly, scalar | poly | `polynomial.mul_scalar` |
| `fhetch.ntt` | NTT1 + NTT2 | `sr_ntt` | poly (coeff) | poly (eval) | `polynomial.ntt` |
| `fhetch.intt` | INTT1 + INTT2 | `sr_intt` | poly (eval) | poly (coeff) | `polynomial.intt` |
| `fhetch.morph` | MORPH1 + MORPH2 | `sr_automorph_*` | poly, `index` (odd, < 2N) | poly | — (no `polynomial` op yet) |

Relinearization and the key switch that accompanies a rotation are *sequences*
of these opcodes (§0.2, `lwe-decompose`), not instructions, so they have no op.

> **Revised (implemented).** The draft below argued for dropping ops that are 1:1
> with `polynomial`. That no longer applies: a `polynomial` op on an RNS
> polynomial is `k` FHETCH instructions, one per limb, so every `fhetch` op is
> justified by being *one* hardware opcode on *one* residue. The emitter (§5)
> therefore emits from `fhetch` ops alone.

Every row of the draft table except `poly_rotate` and `relinearize` had a direct
`polynomial`/`mod_arith` counterpart (`polynomial.mul` on `form=eval` operands,
`polynomial.mul_scalar`, `polynomial.add`, `polynomial.sub`, `polynomial.ntt`,
`polynomial.intt`), so the draft required each `fhetch` op to justify why it is
not just the `polynomial` op — a distinct hardware opcode, a different operand
encoding, or a hardware-only fusion — and concluded that ops 1:1 with
`polynomial` should be dropped. The single-residue operand encoding is exactly
such a difference, which is why the implemented dialect keeps them. The goal
stands: `fhetch` is an API mirror of the external IR, as `new_backend.md`
requires, rather than a shadow copy of HEIR's own arithmetic.

**Attributes & metadata:**

```
// Batch size (parallelism offered by accelerator)
#fhetch.batch_size<8>

// Resource constraints (per-operation)
#fhetch.latency<cycles: i64>
#fhetch.throughput<ops_per_cycle: f64>

// Hardware capabilities annotation
#fhetch.target_hw<name: string>
```

**Type validation & constraints** (implemented; all but `morph`'s are
declarative in ODS):

- Single residue: a type constraint requires a `!mod_arith.int<q>` coefficient
  type and a polynomial modulus, rejecting RNS polynomials
- Same ring and form: `SameOperandsAndResultType` plus the reused
  `SameOperandsAndResultForm` trait on `add`/`sub`/`mul`
- Scalars: `TypesMatchWith` requires the scalar of `addi`/`subi`/`muli` to have
  the ring's coefficient type, i.e. the same modulus
- NTT domain: `ntt` takes `form = coeff` and returns the same ring in
  `form = eval`; `intt` the reverse
- `morph`: a C++ verifier requires an odd `index` in `(0, 2N)`, since
  `x -> x^k` is an automorphism of `Z_q[x]/(x^N + 1)` only for odd `k`
- Folders cover the opcode identities (`addi x, 0`, `muli x, 1`, `muli x, 0`,
  `sub x, x`, `intt(ntt x)`, zero through `ntt`/`intt`/`morph`), reducing scalars
  modulo `q` first so a limb-wise 0 or 1 folds after the RNS split
- Still open: validity of the moduli against the accelerator (§3.3) and batch
  semantics

### 2.2 Testing the Dialect (via lit/FileCheck)

**Test cases** (following Lattigo example, `tests/Dialect/Lattigo/IR/ckks_ops.mlir`).
Per `AGENTS.md`, never use `CHECK-LABEL` in HEIR tests — plain `CHECK` only.

```mlir
// tests/Dialect/Fhetch/IR/ops.mlir (excerpt)
// RUN: heir-opt %s | FileCheck %s

!Zq = !mod_arith.int<65537 : i64>
#ring = #polynomial.ring<coefficientType = !Zq, polynomialModulus = <1 + x**1024>>
!coeff = !polynomial.polynomial<ring = #ring, form = coeff>
!eval = !polynomial.polynomial<ring = #ring, form = eval>

// CHECK: func.func @transforms
func.func @transforms(%a: !coeff) -> !coeff {
  // CHECK: fhetch.ntt
  %0 = fhetch.ntt %a : (!coeff) -> !eval
  // CHECK: fhetch.morph
  // CHECK-SAME: index = 3
  %1 = fhetch.morph %0 {index = 3 : i64} : !eval
  // CHECK: fhetch.intt
  %2 = fhetch.intt %1 : (!eval) -> !coeff
  return %2 : !coeff
}
```

Note the operands are `!polynomial.polynomial` over a single `!mod_arith.int<q>`,
and the NTT domain is expressed by `form = eval` in the type rather than by an
attribute. `verifier.mlir` covers each rejected case and `folding.mlir` the
folders.

**Deliverables:**
- [x] `FhetchDialect.h/.cpp/.td`: dialect registration, type/op verification
- [x] `FhetchOps.td`: only those operations justified per §2.1. FHETCH is
      single-residue, so `add`/`sub`/`mul`/`addi`/`subi`/`muli`/`ntt`/`intt`/
      `morph` are each one ISA opcode on one limb, not a copy of the
      RNS-level `polynomial` op; operands are constrained to
      `!mod_arith.int<q>` coefficients
- [ ] `FhetchTypes.td`: `poly_ref`, key types, hardware attributes. Deferred
      until resource allocation (§4.2.3) and key switching need them
- [x] Test suite `tests/Dialect/Fhetch/IR/` with `FileCheck` validation
- [x] Registration in `tools/heir-opt.cpp`, `tools/heir-lsp.cpp` and `BUILD` deps
- [x] Folders for identities, e.g. `addi x, 0`, `muli x, 1`, `muli x, 0` and
      `intt(ntt x)`, including per-limb scalars that are 0 or 1 modulo one
      prime only

---

## Phase 3: Lowering Passes

### 3.1 Scheme / Polynomial / `fhetch` Lowering Chain

**Scope:** Convert scheme-level operations (CKKS, BGV) to RNS/polynomial to `fhetch`.

> **Revised.** The original draft lowered `ckks.*` straight to `fhetch.*`. That is
> the category-1 shape (what `openfhe`/`lattigo` do) and is wrong for a hardware
> backend — see §0.2. The correct chain routes through `polynomial`, so that the
> cryptosystem is expressed once, in HEIR, and shared by every hardware vendor:
>
> ```
> ckks / bgv
>   └─ ckks-to-lwe            (exists; 7 elementwise ops only — §0.2)
>       └─ lwe-decompose      (exists; key switch + mod down)
>           └─ lwe-to-polynomial   (exists; broken for plaintext ops — §0.4)
>               └─ polynomial-to-fhetch    ← NEW, the only pass this project owns
> ```

**Pass structure:**

```
lib/Dialect/Polynomial/Conversions/
└── PolynomialToFhetch/
    ├── BUILD
    ├── PolynomialToFhetch.h/.cpp   # Main lowering pass
    └── PolynomialToFhetch.td       # Pass tablegen

tests/Dialect/Polynomial/Conversions/polynomial_to_fhetch/
```

This mirrors the existing `lib/Dialect/Polynomial/Conversions/PolynomialToModArith/`,
which is the closest precedent in-tree.

**Prerequisite work, not owned by this pass** (upstream #2886, §0.2):
`ckks.rotate` → Galois automorphism + `lwe.key_switch_inner`, and
`ckks.rescale` → `lwe.mod_down`. Until these land, no CKKS program containing a
rotation or a rescale can reach `fhetch` at all.

### 3.2 Lowering Strategy

**Step 1: Pre-lowering validation & analysis**

Before lowering, analyze the IR. Reuse `--rotation-analysis` (§0.3) rather than
writing a new rotation analysis:

> **Revised (implemented).** The pass itself checks only what makes a lowering
> impossible: a coefficient-form `polynomial.mul` is a negacyclic convolution,
> which `fhetch.mul` (component-wise) cannot express, so the pass rejects it and
> asks for `--convert-polynomial-mul-to-ntt` first. Checking the degree and
> primes against an accelerator is the separate validation pass of §3.3.
> Excerpt of `PolynomialToFhetch.cpp`:

```cpp
void runOnOperation() override {
  MLIRContext* context = &getContext();
  ModuleOp module = getOperation();

  WalkResult walk = module.walk([](MulOp op) {
    auto polyType = cast<PolynomialType>(getElementTypeOrSelf(op.getType()));
    if (polyType.getForm() == Form::EVAL) return WalkResult::advance();
    op.emitOpError() << "must be in evaluation form to lower to `fhetch`; "
                        "run `--convert-polynomial-mul-to-ntt` first";
    return WalkResult::interrupt();
  });
  if (walk.wasInterrupted()) return signalPassFailure();

  LimbTypeConverter typeConverter;
  ConversionTarget target(*context);
  target.addLegalDialect<fhetch::FhetchDialect>();
  target.addIllegalDialect<rns::RNSDialect>();
  target.addIllegalDialect<PolynomialDialect>();
  // Single-residue constants are what `fhetch` folds to, so they stay.
  target.addDynamicallyLegalOp<ConstantOp>([&](ConstantOp op) {
    return typeConverter.isLegal(op.getOperation());
  });
  target.markUnknownOpDynamicallyLegal(
      [&](Operation* op) { return typeConverter.isLegal(op); });

  RewritePatternSet patterns(context);
  // ... (see Step 3 below)
  addStructuralConversionPatterns(typeConverter, patterns, target);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
```

**Step 2: Type conversion**

> **Revised.** With `fhetch` ops taking `!polynomial.polynomial` operands (§2.1),
> most of this pass needs **no type converter at all** — the polynomial types pass
> through unchanged. A converter is needed only where the hardware demands a
> representation HEIR lacks, e.g. materializing `!fhetch.poly_ref` slots during
> resource allocation (§4.2.3).
>
> **Revised again (implemented).** Because `fhetch` is single-residue (§2.2), the
> pass *does* need a converter, a 1:N one: a polynomial over
> `!rns.rns<q_1, ..., q_k>` becomes `k` polynomials over `!mod_arith.int<q_i>`,
> and an RNS scalar becomes `k` `mod_arith` scalars. Function signatures, calls
> and SCF expand to one value per limb (`addStructuralConversionPatterns`, as in
> `TensorToScalars`). Tensors of RNS polynomials have no single-residue form and
> make the pass fail. A `root` on `polynomial.ntt`/`intt` is dropped, since the
> accelerator applies its own twiddles; eval-form constants computed by HEIR
> must therefore agree with the accelerator's root (open: §3.3).

```cpp
// Splits RNS polynomials and scalars into one value per limb, in limb order.
// Containers of RNS values have no single-residue form, so they fail to
// convert, which makes any op that uses one illegal.
class LimbTypeConverter : public TypeConverter {
 public:
  LimbTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion(
        [](ShapedType type,
           SmallVectorImpl<Type>& types) -> std::optional<LogicalResult> {
          if (!isRNS(type)) return std::nullopt;
          return failure();
        });
    addConversion([](rns::RNSType type, SmallVectorImpl<Type>& types) {
      llvm::append_range(types, type.getBasisTypes());
      return success();
    });
    addConversion(
        [](PolynomialType type,
           SmallVectorImpl<Type>& types) -> std::optional<LogicalResult> {
          RingAttr ring = type.getRing();
          auto rnsType = dyn_cast<rns::RNSType>(ring.getCoefficientType());
          if (!rnsType) return std::nullopt;
          if (!ring.getPolynomialModulus()) return failure();
          for (Type basisType : rnsType.getBasisTypes())
            types.push_back(PolynomialType::get(
                type.getContext(),
                RingAttr::get(basisType, ring.getPolynomialModulus()),
                type.getForm()));
          return success();
        });
  }
};
```

A 1:1 converter to `!fhetch.poly_ref` storage slots remains a separate concern of
resource allocation (§4.2.3), applied after this pass.

**Step 3: Operation conversion patterns**

Define lowering for `polynomial` operations to `fhetch` operations.

Note the API: HEIR requires `OpType::create(rewriter, ...)`; the
`rewriter.create<OpType>(...)` form used in the original draft is deprecated
(`AGENTS.md`).

> **Revised (implemented).** Every arithmetic op lowers through one generic
> pattern that emits one `fhetch` op per limb, taking limb `i` of every operand.
> The NTT domain needs no handling here: it is in the *type*, carried over limb
> by limb, and `--convert-polynomial-mul-to-ntt` has already placed the
> transforms.

```cpp
template <typename SourceOp, typename TargetOp>
struct ConvertLimbwise : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SourceOp op,
      typename OpConversionPattern<SourceOp>::OneToNOpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    SmallVector<Type> resultTypes;
    if (failed(
            this->getTypeConverter()->convertType(op.getType(), resultTypes)))
      return rewriter.notifyMatchFailure(op, "unsupported result type");
    for (ValueRange operand : adaptor.getOperands())
      if (operand.size() != resultTypes.size())
        return rewriter.notifyMatchFailure(op, "mismatched limb counts");

    SmallVector<Value> limbs;
    limbs.reserve(resultTypes.size());
    for (auto [i, resultType] : llvm::enumerate(resultTypes)) {
      SmallVector<Value> operands;
      for (ValueRange operand : adaptor.getOperands())
        operands.push_back(operand[i]);
      limbs.push_back(
          TargetOp::create(rewriter, op.getLoc(), resultType, operands));
    }
    rewriter.replaceOpWithMultiple(op, {limbs});
    return success();
  }
};

patterns.add<ConvertLimbwise<AddOp, fhetch::AddOp>,
             ConvertLimbwise<SubOp, fhetch::SubOp>,
             ConvertLimbwise<MulOp, fhetch::MulOp>,
             ConvertLimbwise<MulScalarOp, fhetch::MulIOp>,
             ConvertLimbwise<NTTOp, fhetch::NTTOp>,
             ConvertLimbwise<INTTOp, fhetch::INTTOp>, ...>(typeConverter,
                                                            context);
```

The remaining patterns move limbs without emitting `fhetch` ops:
`polynomial.extract_slice` and `rns.extract_slice` keep a range of limbs,
`rns.extract_residue` picks one, `rns.pack` concatenates, and `rns.constant` and
RNS `polynomial.constant` become one constant per limb.

**Step 4: Handle scheme-specific semantics**

> **Revised.** With the corrected chain (§3.1), scheme-specific semantics are
> resolved *before* this pass runs — by `ckks-to-lwe`, `lwe-decompose`, and
> `lwe-to-polynomial`. By the time IR reaches `polynomial-to-fhetch` there is no
> CKKS scale management or BGV modulus chain left to interpret, only ring
> arithmetic. This is the main simplification the corrected chain buys, and the
> main reason not to lower `ckks` directly.

What *does* remain scheme-adjacent at this level:

- **Scale/level metadata** carried on `mgmt` attributes, if the emitter must
  reproduce it in the FHETCH IR header
- **CGGI** does not route through `polynomial` at all; it lowers via `lwe`
  to bit-level dialects. A CGGI-to-FHETCH story needs a separate design and is
  **out of scope** until the CKKS/BGV path works end to end.

### 3.3 Parameter Selection & Compatibility

**Challenge:** HEIR scheme dialects pre-select RNS parameters; FHETCH hardware may not support those exact parameters.

**Solutions:**

1. **Flexible RNS mapping:** If HEIR uses 4 RNS limbs but hardware prefers 5, validate that the computation is still correct (extra prime doesn't hurt, just wastes bandwidth).

2. **Parameter negotiation pass:** Insert an analysis pass that:
   - Checks HEIR's selected RNS primes against hardware database
   - Inserts warning/error if incompatible
   - Optionally applies `rns-reparameterize` pass to adjust

3. **Hardware capability metadata** (proposed, not implemented): since every
   `fhetch` op is single-residue, the degree and modulus are already in its
   type, so what remains is target-level and fits a module attribute rather
   than one per op:
   ```mlir
   module attributes {fhetch.target = #fhetch.target<
       name = "niobium_accelerator_v1", max_degree = 65536>} {
     ...
     %0 = fhetch.mul %lhs, %rhs : !eval
   }
   ```

### 3.4 Deliverables

- [x] `PolynomialToFhetch.cpp`: main dialect conversion pass (**not** `LWEToFhetch`)
- [x] Opcode converters: `polynomial`/`mod_arith` ops to `fhetch` opcodes
- [ ] Parameter analysis and validation pass
- [x] Test suite: conversion correctness via IR matching with `FileCheck`
- [ ] Tensors of RNS polynomials: `lwe-to-polynomial` emits ciphertexts as
      `tensor<2x!poly>`, which the pass rejects today; split them limb-wise
      (or run `--tensor-to-scalars` first) before Phase 6 can feed it
- [ ] NTT twiddles: the pass drops `polynomial.ntt`'s `root`; check that the
      accelerator's primitive root matches HEIR's, or reject eval-form
      constants whose root differs
- [ ] *Blocked on upstream:* `ckks.rotate`/`ckks.rescale` lowering (#2886)
- [ ] *Not blocked by* the `lwe-to-polynomial` plaintext bug (§0.4), which is a
      separate track: the pass consumes `polynomial` IR and is tested from
      hand-written input; the bug only gates feeding it from `ckks`/`bgv`
      (Phases 6–7)

---

## Phase 4: Backend-Specific Optimization Passes

### 4.1 Fhetch Dialect Transforms

**Operations to optimize in fhetch IR:**

```
lib/Dialect/Fhetch/Transforms/
├── BUILD
├── Passes.h / Passes.td           # Pass registry (tablegen, per HEIR convention)
├── RotationScheduling.cpp         # Optimize rotation/Galois automorphism order
├── MemoryCoalescing.cpp           # Group polynomials for memory efficiency
├── ResourceAllocation.cpp         # Register pressure balancing (as needed)
└── Configuration.cpp              # Setup / key generation insertion

tests/Dialect/Fhetch/Transforms/   # pass-specific tests
```

`NTTScheduling.cpp` and `CSEOnFhetch.cpp` are **dropped** from the original list;
see §4.2.1 and §4.2.4.

### 4.2 Key Optimizations

#### 4.2.1 NTT/INTT Reduction — *already implemented upstream*

**Problem:** Frequent switching between canonical and NTT domains incurs unnecessary transforms.

> **Revised.** HEIR already does this at the `polynomial` level, and doing it
> there is strictly better: it happens before the hardware lowering, so every
> backend benefits.
>
> - `--convert-polynomial-mul-to-ntt` — "Optimally inserts NTT/INTT into a
>   Polynomial IR". This is precisely the pass drafted below.
> - `INTTAfterNTT` / `NTTAfterINTT` in `PolynomialCanonicalization.td` — eliminate
>   `intt(ntt(x))` and `ntt(intt(x))` round trips, subject to matching roots.
> - `--attach-ntt-roots` — attaches the appropriate roots of unity.
>
> The exact before/after transformation sketched in the original draft:
>
> ```
> // Before:                          // After:
> %0 = ntt(%a)                        %0 = ntt(%a)
> %1 = mul(%0, %0)                    %1 = mul(%0, %0)
> %2 = intt(%1)                       %4 = mul(%1, %1)   // %1 stays in NTT
> %3 = ntt(%2)                        %5 = intt(%4)      // single INTT at end
> %4 = mul(%3, %3)
> %5 = intt(%4)
> ```
>
> is what the `INTTAfterNTT` canonicalization performs on `%2`/`%3`.
>
> **Action:** run these passes in the pipeline (§6.1) before
> `polynomial-to-fhetch`. Write a new pass **only** if profiling shows the
> generic placement is wrong for FHETCH hardware — and if so, contribute the
> improvement upstream to `--convert-polynomial-mul-to-ntt` rather than forking
> it into `fhetch`.

#### 4.2.2 Rotation Scheduling

**Problem:** Galois automorphisms/rotations are expensive. Ordering matters for the hardware.

**Solution:** Analyze rotation dependency graph and reorder to maximize hardware locality.

Reuse HEIR's existing `RotationAnalysis` (`--rotation-analysis`, which annotates
the module with discovered indices) for collection; this pass owns only the
hardware-specific *scheduling*. Related prior art worth reading before writing
this: `--optimize-relinearization`, `--openfhe-hoist-key-switch-down`, and
`--openfhe-fast-rotation-precompute`.

```cpp
struct RotationScheduling
    : impl::RotationSchedulingBase<RotationScheduling> {
  using RotationSchedulingBase::RotationSchedulingBase;

  void runOnOperation() override {
    // 1. Collect rotations and their dependencies (reuse RotationAnalysis).
    //    At this level a rotation is a `fhetch.morph` per limb plus the
    //    key-switch sequence that follows it.
    SmallVector<fhetch::MorphOp> rotations;
    getOperation().walk([&](fhetch::MorphOp op) { rotations.push_back(op); });

    // 2. Detect rotations that can be batched or composed.
    RotationGroupAnalysis rga(rotations);

    // 3. Schedule in hardware-friendly order (minimize key switching,
    //    cache misses, etc.) and reorder.
    for (auto rot : rga.computeOptimalSchedule())
      rot->moveBefore(/* ... */);
  }
};
```

#### 4.2.3 Resource Allocation

**Problem:** Hardware accelerators have limited polynomial registers. Must spill or fill.

**Solution:** Allocate virtual `!fhetch.poly_ref` slots and insert load and store
opcodes. Both the type and the `fhetch.poly_load`/`fhetch.poly_store` ops are new,
to be added with this pass (§2.2 defers the types until then).

Use MLIR's own `Liveness` analysis (`mlir/Analysis/Liveness.h`) rather than a
hand-rolled one.

```cpp
struct ResourceAllocation
    : impl::ResourceAllocationBase<ResourceAllocation> {
  using ResourceAllocationBase::ResourceAllocationBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Liveness liveness(func);

    // If the live polynomial count exceeds the hardware register file,
    // insert fhetch.poly_store / fhetch.poly_load spills.
    for (Block& block : func) {
      for (Operation& op : block) {
        if (countLivePolynomials(liveness, &op) > hwConfig.numPolyRegs)
          insertSpill(&op, /* ... */);
      }
    }
  }
};
```

#### 4.2.4 CSE on Fhetch Ops — *likely unnecessary*

> **Revised.** The drafted pass merges two `poly_scale` ops with the same operand
> and modulus. Generic MLIR `--cse` already does exactly that for any op marked
> `Pure` (or `NoMemoryEffect`) with equal operands and attributes — and the
> drafted NTT-domain multiply op was already marked `NoMemoryEffect`.
>
> **Action:** mark `fhetch` ops with the correct effect traits and rely on
> `--cse`. Revisit only for genuinely non-generic algebraic identities (e.g.
> fusing two successive `poly_scale`s into one by multiplying their moduli),
> which is a canonicalization pattern on the op, not a standalone pass.
>
> **Revised (implemented).** Every `fhetch` op is `Pure`, so `--cse` applies, and
> the identities with a constant operand are folders (§2.1). Still open: fusing
> `muli(muli(x, a), b)` into `muli(x, a·b mod q)` and `addi` chains likewise, as
> canonicalization patterns.

### 4.3 Configuration Pass

**Goal:** Analyze the IR and insert function calls for hardware setup: key generation, etc.

Model this on `lib/Dialect/Lattigo/Transforms/ConfigureCryptoContext.cpp`, which
solves the same problem for Lattigo. Check the `key_mgmt` dialect first — HEIR
may already carry the key material this pass would otherwise synthesize.

```cpp
struct FhetchConfiguration
    : impl::FhetchConfigurationBase<FhetchConfiguration> {
  using FhetchConfigurationBase::FhetchConfigurationBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // 1. Extract security parameters, polynomial degree, RNS structure.
    //    Degree and RNS structure come from the RingAttr of any polynomial
    //    operand; scheme params from the module's #ckks.scheme_param attribute.
    auto params = extractParameters(func);

    // 2. Scan for operations that require extra key material.
    bool needsBootstrap = func.walk([](fhetch::BootstrapOp op) {
      return WalkResult::interrupt();
    }).wasInterrupted();

    // 3. Insert configuration operations at function entry.
    ImplicitLocOpBuilder b(func.getLoc(), &func.front().front());
    fhetch::SetupOp::create(b, params, needsBootstrap);

    if (needsRotations)
      fhetch::GenRotationKeysOp::create(b, rotations);
  }
};
```

### 4.4 Deliverables

- [ ] `Passes.td`: pass registry (tablegen, not a hand-written `.cpp` registry)
- [ ] ~~`NTTSchedulingPass.cpp`~~ — dropped, use `--convert-polynomial-mul-to-ntt` (§4.2.1)
- [ ] `RotationScheduling.cpp`: Galois scheduling
- [ ] `ResourceAllocation.cpp`: register pressure, spilling
- [ ] ~~`FhetchCSE.cpp`~~ — dropped, use generic `--cse` plus effect traits (§4.2.4)
- [ ] `Configuration.cpp`: setup, key generation
- [ ] Test suite for each pass

---

## Phase 5: Code Generation (heir-translate)

### 5.1 FHETCH IR Emitter

**Scope:** Convert `fhetch` dialect IR to FHETCH Polynomial IR serialization.

**Structure:**

Follow `lib/Target/Lattigo/LattigoEmitter.cpp`, the closest in-tree precedent.

```
lib/Target/Fhetch/
├── BUILD
├── FhetchEmitter.h/.cpp          # Main emitter
└── IRSerializer.h/.cpp           # Serialize to FHETCH IR format

tests/Target/Fhetch/emit_fhetch.mlir
```

### 5.2 Emission Strategy

**FHETCH IR is likely a binary or text format.** Emit operations in order:

> **Revised.** The draft had the emitter handle `polynomial` ops directly, on the
> premise that ops 1:1 with FHETCH opcodes never get a `fhetch` counterpart. With
> the single-residue dialect (§2.1), every FHETCH instruction *is* a `fhetch` op,
> so the emitter consumes `fhetch` ops only: one op, one opcode, one modulus,
> taken from the operand type. Besides `fhetch` ops, it sees only
> `polynomial.constant` and `mod_arith.constant` (the immediates) and `func`
> structure.

Use `llvm::TypeSwitch`, as `LattigoEmitter` does, rather than an `if`/`else if`
chain.

Note `getAsOpaquePointer()` in the original draft would emit raw host addresses —
non-deterministic and untestable with `FileCheck`. Use the `ValueLayout` id map
from §5.3 instead.

```cpp
class FhetchEmitter {
 public:
  LogicalResult emitOperation(Operation* op) {
    return llvm::TypeSwitch<Operation*, LogicalResult>(op)
        .Case<fhetch::AddOp, fhetch::SubOp, fhetch::MulOp>(
            [&](auto op) { return printBinary(op); })
        .Case<fhetch::AddIOp, fhetch::SubIOp, fhetch::MulIOp>(
            [&](auto op) { return printScalar(op); })
        .Case<fhetch::NTTOp, fhetch::INTTOp, fhetch::MorphOp>(
            [&](auto op) { return printOperation(op); })
        .Case<polynomial::ConstantOp, mod_arith::ConstantOp>(
            [&](auto op) { return printConstant(op); })
        .Default([&](Operation* op) {
          return op->emitOpError("unsupported in the FHETCH emitter");
        });
  }

 private:
  // The modulus of every FHETCH instruction comes from its operand type.
  static APInt getModulus(Type type) {
    auto ring = cast<polynomial::PolynomialType>(type).getRing();
    return cast<mod_arith::ModArithType>(ring.getCoefficientType())
        .getModulus()
        .getValue();
  }

  template <typename OpTy>
  LogicalResult printBinary(OpTy op) {
    os_ << getOpcode(op) << " %" << layout_.allocateId(op.getResult())
        << ", %" << layout_.allocateId(op.getLhs())
        << ", %" << layout_.allocateId(op.getRhs())
        << ", q=" << getModulus(op.getType()) << "\n";
    return success();
  }

  raw_ostream& os_;
  ValueLayout layout_;
};
```

**Possible output format examples:**

```
// Text format (human-readable), one line per limb-level instruction; a two-limb
// RNS product after --polynomial-to-fhetch:
SETUP { degree=2048, moduli=[q0, q1] }
NTT  %4, %0, q=q0
NTT  %5, %1, q=q1
MUL  %6, %4, %2, q=q0
MUL  %7, %5, %3, q=q1
INTT %8, %6, q=q0
INTT %9, %7, q=q1
MULI %10, %8, 4503599627370496, q=q0
...
RETURN %10, %11
```

`NTT`, `INTT` and `MORPH` stand for the instruction pairs NTT1/NTT2,
INTT1/INTT2 and MORPH1/MORPH2. The actual encoding (text, binary with
fixed-width opcodes, or calls to the `fhetch::sr_*` API) is still to be taken
from the FHETCH specification (§1.1, operand encoding).

### 5.3 Value Mapping & Resource Layout

**Challenge:** FHETCH operations reference polynomials by identifiers. Must map MLIR SSA values to hardware polynomial registers or memory.

```cpp
struct ValueLayout {
  DenseMap<Value, uint32_t> value_to_id;  // SSA value → polynomial ID
  uint32_t next_id = 0;

  uint32_t allocateId(Value v) {
    if (!value_to_id.count(v)) {
      value_to_id[v] = next_id++;
    }
    return value_to_id[v];
  }
};
```

### 5.4 Testing Emission

**Test via `FileCheck`:**

```mlir
// tests/Target/Fhetch/emit_fhetch.mlir
// RUN: heir-translate --emit-fhetch %s | FileCheck %s

!Zq = !mod_arith.int<35184372744193 : i64>
#ring = #polynomial.ring<coefficientType = !Zq, polynomialModulus = <1 + x**2048>>
!poly = !polynomial.polynomial<ring = #ring, form = coeff>
!poly_ntt = !polynomial.polynomial<ring = #ring, form = eval>

func.func @dot_product(%lhs: !poly, %rhs: !poly) -> !poly {
  %lhs_ntt = fhetch.ntt %lhs : (!poly) -> !poly_ntt
  %rhs_ntt = fhetch.ntt %rhs : (!poly) -> !poly_ntt
  %mult = fhetch.mul %lhs_ntt, %rhs_ntt : !poly_ntt
  %result = fhetch.intt %mult : (!poly_ntt) -> !poly
  return %result : !poly
}

// CHECK: SETUP
// CHECK: NTT
// CHECK: NTT
// CHECK: MUL
// CHECK: INTT
// CHECK: RETURN
```

The input is single-residue `fhetch` IR, as `--polynomial-to-fhetch` produces it;
an RNS program reaches the emitter already split into one op per limb. Per
`AGENTS.md`, no `CHECK-LABEL`.

### 5.5 Deliverables

- [ ] `FhetchEmitter.h/.cpp`: operation emission logic
- [ ] `IRSerializer.h/.cpp`: FHETCH IR serialization
- [ ] Test suite with expected output validation

---

## Phase 6: Pipeline Definition

### 6.1 New Pipeline: `scheme-to-fhetch`

**Location:** `lib/Pipelines/ArithmeticPipelineRegistration.cpp`

```cpp
struct SchemeToFhetchPipelineOptions
    : public PassPipelineOptions<SchemeToFhetchPipelineOptions> {
  // Hardware target selection
  Option<std::string> target{
    *this, "target",
    llvm::cl::desc("Target hardware (e.g., 'niobium_v1', 'optalysys')"),
    llvm::cl::init("generic")
  };

  // Optimization level
  Option<int> optimization_level{
    *this, "opt-level",
    llvm::cl::desc("Optimization level (0=none, 1=basic, 2=aggressive)"),
    llvm::cl::init(1)
  };

  // Scheme selection
  Option<std::string> scheme{
    *this, "scheme",
    llvm::cl::desc("FHE scheme (ckks, bgv, cggi)"),
    llvm::cl::required()
  };
};

void addSchemeToFhetchPipeline(OpPassManager& pm,
                               const SchemeToFhetchPipelineOptions& opts) {
  // 1. Annotate module with backend & scheme info.
  pm.addPass(createAnnotateModulePass(opts.scheme, "fhetch"));

  // 2. Scheme dialect -> lwe. These are existing HEIR passes; note they do NOT
  //    yet handle ckks.rotate / ckks.rescale (see 0.2 -- upstream #2886).
  if (opts.scheme == "ckks") {
    pm.addPass(ckks::createDecomposeRelinearize());
    pm.addPass(ckks::createCKKSToLWE());
  } else if (opts.scheme == "bgv") {
    pm.addPass(bgv::createBGVToLWE());
  }

  // 3. Decompose the remaining cryptosystem primitives: key switching
  //    (lwe.key_switch_inner) and modulus switching (lwe.mod_down).
  pm.addPass(lwe::createDecomposeLWEOps());

  // 4. lwe -> polynomial, with rns-of-mod_arith coefficients. Ciphertexts come
  //    out as tensor<2x!poly>, which polynomial-to-fhetch cannot split yet;
  //    unroll them to scalars first (§3.4, open item).
  pm.addPass(lwe::createLWEToPolynomial());
  pm.addPass(createTensorToScalars());

  // 5. Place NTT/INTT optimally BEFORE lowering to the hardware dialect; the
  //    lowering rejects coefficient-form products. Canonicalization folds
  //    intt(ntt(x)) round trips. Roots attached by --attach-ntt-roots are
  //    dropped by the lowering, since the accelerator uses its own (§3.4).
  pm.addPass(polynomial::createConvertPolynomialMulToNTT());
  pm.addPass(createCanonicalizerPass());

  // 6. polynomial -> fhetch, one op per RNS limb. The only lowering this
  //    project owns; it takes no options. Canonicalize again so the `fhetch`
  //    folders see the per-limb constants.
  pm.addPass(polynomial::createPolynomialToFhetch());
  pm.addPass(createCanonicalizerPass());

  // 7. Backend-specific optimizations. NTT scheduling and a bespoke CSE are
  //    deliberately absent: see 4.2.1 and 4.2.4.
  if (opts.optimization_level >= 1) {
    pm.addPass(fhetch::createRotationScheduling());
    pm.addPass(createCSEPass());
  }

  if (opts.optimization_level >= 2) {
    pm.addPass(fhetch::createResourceAllocation(opts.target));
  }

  // 8. Insert configuration operations.
  pm.addPass(fhetch::createFhetchConfiguration(opts.target));

  // 9. Dead code elimination.
  pm.addPass(createSymbolDCEPass());
}

// Register the pipeline, following the existing registrations in this file
// (e.g. the lattigo pipeline).
void registerSchemeToFhetchPipeline() {
  PassPipelineRegistration<SchemeToFhetchPipelineOptions>(
    "scheme-to-fhetch",
    "Lowering from FHE scheme dialect to FHETCH Polynomial IR",
    addSchemeToFhetchPipeline
  );
}
```

The `cggi` option is listed in `scheme` above but is **out of scope** for the
first release: CGGI does not route through `polynomial` (§3.2 Step 4). Either
drop it from the option or reject it with a clear diagnostic.

### 6.2 CLI Integration

**Update** `tools/heir-opt.cpp`:

```cpp
int main(int argc, char **argv) {
  registerAllDialects(registry);

  // ... existing registrations ...

  // Register new FHETCH pipeline
  registerSchemeToFhetchPipeline();

  // ...
}
```

**Usage:**

```bash
heir-opt input.mlir \
  --pass-pipeline="builtin.module(scheme-to-fhetch{scheme=ckks target=niobium_v1 opt-level=2})" \
  -o output.mlir
```

### 6.3 Deliverables

- [ ] `SchemeToFhetchPipelineOptions` struct definition
- [ ] `addSchemeTofhetchPipeline()` implementation
- [ ] Pipeline registration in `heir-opt.cpp`
- [ ] Documentation: pipeline flags and usage

---

## Phase 7: End-to-End Testing Infrastructure

### 7.1 E2E Test Strategy

**Testing flow:**

```
MLIR (CKKS/BGV)
    ↓
HEIR compiler (scheme-to-fhetch pipeline)
    ↓
FHETCH Polynomial IR
    ↓
[Validation: IR correctness & syntax]
    ↓
Niobium FHETCH Compiler
    ↓
Niobium accelerator code
    ↓
Niobium simulator / QEMU / FPGA
    ↓
Functional correctness validation
```

**Test infrastructure:**

```
tests/Examples/fhetch/
├── ckks/
│   ├── dot_product_8f/
│   │   ├── BUILD
│   │   ├── dot_product_8f.mlir
│   │   ├── dot_product_8f_harness.cpp
│   │   └── dot_product_8f_test.cpp
│   ├── matmul_4x4/
│   └── ...
├── bgv/
│   ├── ...
├── common/
│   ├── BUILD
│   └── fhetch_test.bzl               (BUILD macro for FHETCH tests)
├── validation/
│   ├── ir_syntax_validator.cpp       (Validate FHETCH IR format)
│   └── ir_semantics_validator.cpp    (Validate operations make sense)
└── fixtures/
    ├── expected_outputs/
    └── test_keys/
```

### 7.2 Niobium Compiler Integration

**Dependency: Niobium FHETCH Accelerator Compiler**

Add to `MODULE.bazel`:

```starlark
bazel_dep(name = "nbcc", version = "1.0")
```

**OR if not yet in Bazel Central Registry:** locally via `local_path_override` during development.

**Wrapper script for Niobium compiler** (in `tools/`):

```bash
# tools/nbcc.sh
#!/bin/bash
# Wrapper: FHETCH IR → Niobium accelerator code

FHETCH_IR_FILE=$1
OUTPUT_DIR=$2
NIOBIUM_COMPILER=${NIOBIUM_COMPILER:-nbcc}

# Validate FHETCH IR format (optional pre-check)
# $FHETCH_VALIDATOR $FHETCH_IR_FILE || exit 1

# Compile via Niobium (TODO)
$NIOBIUM_COMPILER \
  --input-fhetch-ir $FHETCH_IR_FILE \
  --output-dir $OUTPUT_DIR \
  --target-device niobium_accelerator \
  || exit 1

# Report success
echo "Successfully compiled to Niobium accelerator code in $OUTPUT_DIR"
```

**Bazel rule for compilation:**

```starlark
# build/fhetch_rules.bzl

def nbcc(name, fhetch_ir, **kwargs):
    """Compile FHETCH IR using Niobium compiler."""

    native.genrule(
        name = name + "_compile",
        srcs = [fhetch_ir],
        outs = [name + "_niobium.h"],
        cmd = """
            $(location //tools:nbcc) $< $(GENDIR) && \
            cp $(GENDIR)/niobium_code.h $@
        """,
        tools = [
            "//tools:nbcc",
            "@nbcc//:compiler_binary",
        ],
        **kwargs
    )
```

### 7.3 FHETCH IR Validation Layer

**Before passing to Niobium, validate FHETCH IR locally:**

Create lightweight validators that check:
1. **Syntax:** Operations, types, attributes are well-formed
2. **Semantics:** NTT domain consistency, polynomial degree matches, RNS structure is coherent

> **Scope reduction.** Checks (1) and the NTT/degree/RNS parts of (2) are
> **already enforced inside MLIR** once the dialect is defined correctly: op
> verifiers cover syntax, `SameOperandsAndResultForm` covers NTT-domain
> consistency, and `AllTypesMatch`/`SameOperandsAndResultRings` cover degree and
> RNS agreement (§2.1). `heir-opt` fails the build if any of them is violated,
> so a separate validator that re-checks them adds maintenance cost and no
> coverage.
>
> Restrict this layer to what MLIR *cannot* check: conformance of the **emitted
> FHETCH IR text/binary** to the published spec — opcode encoding, operand
> widths, header fields. That validator reads the emitter's output, not MLIR IR.

```cpp
// tests/Examples/fhetch/validation/ir_semantics_validator.cpp
// Operates on the *emitted* FHETCH IR, after heir-translate.

class FhetchIRValidator {
  // NOTE: NTT-domain consistency is NOT checked here; the polynomial type's
  // `form` parameter and the SameOperandsAndResultForm trait guarantee it
  // upstream, and the emitter cannot produce a violating program.

  bool validatePolynomialDegree(const FhetchProgram& prog) {
    // Rule: all polynomials in a computation must have matching degree
    std::optional<int> expected_degree;
    for (auto& poly : prog.polynomials()) {
      if (!expected_degree) {
        expected_degree = poly.degree();
      } else if (poly.degree() != expected_degree.value()) {
        logError("Polynomial degree mismatch: %d vs %d",
                 poly.degree(), expected_degree.value());
        return false;
      }
    }
    return true;
  }

  bool validateRNSStructure(const FhetchProgram& prog) {
    // Rule: all polynomials must have identical RNS limb count & prime set
    std::optional<std::vector<uint64_t>> expected_primes;
    for (auto& poly : prog.polynomials()) {
      if (!expected_primes) {
        expected_primes = poly.rnsLimbs();
      } else if (poly.rnsLimbs() != expected_primes.value()) {
        logError("RNS structure mismatch");
        return false;
      }
    }
    return true;
  }
};
```

**Integrate validation into test pipeline:**

```starlark
def fhetch_test(name, mlir_src, heir_opt_flags, srcs):
    """Create FHETCH → Niobium e2e test."""

    # 1. Compile MLIR to FHETCH IR
    native.genrule(
        name = name + "_emit_fhetch",
        srcs = [mlir_src],
        outs = [name + "_fhetch.ir"],
        cmd = """
            $(location //tools:heir-opt) $< \\
                --pass-pipeline='builtin.module($(HEIR_OPT_FLAGS))' \\
                | $(location //tools:heir-translate) --emit-fhetch \\
                > $@
        """,
        tools = [
            "//tools:heir-opt",
            "//tools:heir-translate",
        ],
    )

    # 2. Validate FHETCH IR
    native.sh_test(
        name = name + "_validate_ir",
        srcs = ["//tests/Examples/fhetch/validation:ir_validator.sh"],
        args = [
            "$(location :" + name + "_emit_fhetch)",
        ],
        data = [
            ":" + name + "_emit_fhetch",
            "//tests/Examples/fhetch/validation:ir_semantics_validator",
        ],
    )

    # 3. Compile FHETCH IR via Niobium
    nbcc(
        name = name + "_niobium_code",
        fhetch_ir = name + "_emit_fhetch",
    )

    # 4. Run functional test with Niobium-compiled code
    native.cc_test(
        name = name,
        srcs = srcs,
        deps = [
            ":" + name + "_niobium_code",
            "//tests/Examples/fhetch:niobium_harness",
            "@com_google_googletest//:gtest_main",
        ],
        data = [
            ":test_keys.json",  # Pre-generated test encryption keys
        ],
    )
```

### 7.4 Test Harness: Niobium Code Execution

**Simplified harness:** Just call Niobium-compiled functions, don't simulate FHE.

```cpp
// tests/Examples/fhetch/niobium_harness.h

class NiobiumTestHarness {
  // Pre-loaded test keys (generated offline, checked into repo)
  struct TestKeys {
    std::vector<uint8_t> public_key;
    std::vector<uint8_t> private_key;
    std::vector<uint8_t> rotation_keys;
  };

  TestKeys test_keys_;

  // Load test keys from JSON fixture
  void loadTestKeys(const std::string& json_path) {
    auto json = parseJSON(json_path);
    test_keys_.public_key = base64_decode(json["public_key"]);
    test_keys_.private_key = base64_decode(json["private_key"]);
    test_keys_.rotation_keys = base64_decode(json["rotation_keys"]);
  }

  // Encrypt plaintext using test key
  std::vector<uint8_t> encrypt(const std::vector<double>& plaintext) {
    // Call OpenFHE or Lattigo to encrypt with test_keys_.public_key
    // Return serialized ciphertext bytes
    return encrypt_with_key(plaintext, test_keys_.public_key);
  }

  // Decrypt result using test key
  std::vector<double> decrypt(const std::vector<uint8_t>& ciphertext) {
    // Call OpenFHE or Lattigo to decrypt with test_keys_.private_key
    // Return plaintext
    return decrypt_with_key(ciphertext, test_keys_.private_key);
  }

  // Call Niobium-compiled computation
  std::vector<uint8_t> compute(
      const std::string& func_name,
      const std::vector<std::vector<uint8_t>>& ciphertext_args) {

    // Look up compiled function (generated by Niobium compiler)
    auto func = lookup_niobium_function(func_name);

    // Call it with ciphertext arguments
    return func(ciphertext_args, test_keys_.public_key);
  }
};
```

### 7.5 Example Test: Dot Product

```cpp
// tests/Examples/fhetch/ckks/dot_product_8f/dot_product_8f_test.cpp

#include <gtest/gtest.h>
#include "tests/Examples/fhetch/niobium_harness.h"

// Generated by Niobium compiler
#include "dot_product_8f_niobium.h"

class DotProduct8fTest : public ::testing::Test {
 protected:
  void SetUp() override {
    harness_.loadTestKeys(
        "tests/Examples/fhetch/fixtures/test_keys.json");
  }

  NiobiumTestHarness harness_;
};

TEST_F(DotProduct8fTest, BasicComputation) {
  // 1. Test input vectors
  std::vector<double> input_a = {0.1, 0.2, 0.3, 0.4,
                                  0.5, 0.6, 0.7, 0.8};
  std::vector<double> input_b = {0.2, 0.3, 0.4, 0.5,
                                  0.6, 0.7, 0.8, 0.9};

  // 2. Encrypt inputs
  auto ct_a = harness_.encrypt(input_a);
  auto ct_b = harness_.encrypt(input_b);

  // 3. Call Niobium-compiled dot_product function
  auto result_ct = harness_.compute("dot_product_8f", {ct_a, ct_b});

  // 4. Decrypt and validate
  auto result = harness_.decrypt(result_ct);

  double expected = 2.50;  // dot(input_a, input_b)
  EXPECT_NEAR(result[0], expected, 0.0001)
      << "Dot product mismatch";
}

TEST_F(DotProduct8fTest, VectorOfZeros) {
  std::vector<double> zeros(8, 0.0);
  std::vector<double> nonzeros = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

  auto ct_zeros = harness_.encrypt(zeros);
  auto ct_nonzeros = harness_.encrypt(nonzeros);

  auto result_ct = harness_.compute("dot_product_8f", {ct_zeros, ct_nonzeros});
  auto result = harness_.decrypt(result_ct);

  EXPECT_NEAR(result[0], 0.0, 0.0001);
}

TEST_F(DotProduct8fTest, LargeValues) {
  std::vector<double> large_a(8);
  std::vector<double> large_b(8);
  std::iota(large_a.begin(), large_a.end(), 1000.0);
  std::iota(large_b.begin(), large_b.end(), 2000.0);

  auto ct_a = harness_.encrypt(large_a);
  auto ct_b = harness_.encrypt(large_b);

  auto result_ct = harness_.compute("dot_product_8f", {ct_a, ct_b});
  auto result = harness_.decrypt(result_ct);

  // Compute expected value
  double expected = 0.0;
  for (size_t i = 0; i < 8; i++) {
    expected += large_a[i] * large_b[i];
  }

  EXPECT_NEAR(result[0], expected, expected * 0.001);
}
```

### 7.6 Test Keys & Fixtures

**Pre-generate test keys once, commit to repo:**

```bash
# Generate test keys (run once)
bazel run //tests/Examples/fhetch:gen_test_keys -- \
  --output tests/Examples/fhetch/fixtures/test_keys.json \
  --scheme ckks \
  --degree 2048 \
  --rns_limbs 4
```

**Store as JSON fixture:**

```json
{
  "scheme": "ckks",
  "degree": 2048,
  "rns_limbs": 4,
  "public_key": "base64-encoded...",
  "private_key": "base64-encoded...",
  "rotation_keys": "base64-encoded..."
}
```

### 7.7 BUILD File for Tests

```starlark
# tests/Examples/fhetch/common/fhetch_test.bzl

load("//build:fhetch_rules.bzl", "niobium_compile_fhetch")

def fhetch_test(name, mlir_src, heir_opt_flags, srcs, **kwargs):
    """
    Create a FHETCH e2e test: MLIR → HEIR → FHETCH IR → Niobium → test

    Args:
      name: Test name
      mlir_src: MLIR source file
      heir_opt_flags: heir-opt pipeline flags
      srcs: C++ test harness sources
    """

    # Step 1: Emit FHETCH IR
    native.genrule(
        name = name + "_emit_fhetch",
        srcs = [mlir_src],
        outs = [name + "_fhetch.ir"],
        cmd = """
            $(location //tools:heir-opt) $< \\
                --pass-pipeline='builtin.module($(HEIR_OPT_FLAGS))' \\
                | $(location //tools:heir-translate) --emit-fhetch \\
                > $@
        """,
        tools = [
            "//tools:heir-opt",
            "//tools:heir-translate",
        ],
    )

    # Step 2: Validate IR
    native.sh_test(
        name = name + "_ir_valid",
        srcs = ["//tests/Examples/fhetch/validation:validate.sh"],
        args = ["$(location :" + name + "_emit_fhetch)"],
        data = [":" + name + "_emit_fhetch"],
    )

    # Step 3: Compile via Niobium
    nbcc(
        name = name + "_niobium",
        fhetch_ir = ":" + name + "_emit_fhetch",
    )

    # Step 4: Run functional test
    native.cc_test(
        name = name,
        srcs = srcs,
        deps = [
            ":" + name + "_niobium",
            "//tests/Examples/fhetch:niobium_harness",
            "@com_google_googletest//:gtest_main",
        ],
        data = ["//tests/Examples/fhetch/fixtures:test_keys"],
        **kwargs
    )

# Usage example:
# BUILD file in tests/Examples/fhetch/ckks/dot_product_8f/

load("@heir//tests/Examples/fhetch:common/fhetch_test.bzl", "fhetch_test")

fhetch_test(
    name = "dot_product_8f_test",
    mlir_src = "@heir//tests/Examples/common:dot_product_8f.mlir",
    heir_opt_flags = [
        "--annotate-module=backend=fhetch scheme=ckks",
        "--mlir-to-ckks=ciphertext-degree=2048 first-mod-bits=0",
        "--scheme-to-fhetch",
    ],
    srcs = ["dot_product_8f_test.cpp"],
)
```

### 7.8 Deliverables

- [ ] `fhetch_test.bzl` macro (MLIR to FHETCH IR to Niobium compilation)
- [ ] `niobium_harness.h` (simplified test harness for Niobium-compiled code)
- [ ] `ir_semantics_validator.cpp` (pre-Niobium validation)
- [ ] Niobium compiler Bazel integration (wrapper, BUILD rules)
- [ ] Example tests: `dot_product_8f`, `matmul_4x4`, `bootstrap` (one per scheme)
- [ ] Test fixture: `test_keys.json` (pre-generated encryption keys)
- [ ] Documentation: how to run tests locally vs. CI

---

## Phase 8: Documentation & Standardization

### 8.1 HEIR Documentation

**Add to** `docs/guides/`:

```
docs/guides/
├── new_backend.md (existing)
└── fhetch_integration_guide.md (new)
    ├── Overview: FHETCH in HEIR
    ├── Architecture: dialect layering
    ├── Type system: polynomial types
    ├── Passes: lowering & optimization
    ├── Codegen: IR emission
    ├── Testing: E2E examples
    └── FAQ: common integration questions
```

### 8.2 FHETCH Hardware Integration Guide

**Deliverable: white paper on HEIR → FHETCH for hardware vendors**

Topics:
- HEIR's compilation pipeline and how FHETCH fits
- Expected FHETCH IR format (semantics, constraints)
- Hardware implementer checklist (what you need to support)
- Performance profiling: how to measure & optimize
- Example: Niobium accelerator integration

### 8.3 API Reference

**Document FHETCH dialect operations:**

| Operation | Syntax | Semantics | Latency | Notes |
|-----------|--------|-----------|---------|-------|
| `fhetch.ntt` | `fhetch.ntt %a : (!coeff) -> !eval` | Forward NTT on one residue | HW-specific | Must support every prime the program uses |
| `fhetch.mul` | `fhetch.mul %a, %b : !poly` | Component-wise product mod q | HW-specific | A ring product only in `form = eval` |
| ... | ... | ... | ... | ... |

The op descriptions in `FhetchOps.td` are the source for this table; HEIR's
dialect documentation generates from them.

### 8.4 Design Document: HEIR and FHETCH Contract

**Clarify assumptions & boundaries:**

- **What HEIR guarantees:** All types are well-formed, RNS structure is consistent, rotations are listed
- **What hardware must guarantee:** Support all operations, correct RNS arithmetic, Galois key availability
- **Error handling:** Unsupported operations, overflow/underflow semantics
- **Performance hints:** Annotations for latency-critical operations

### 8.5 Deliverables

- [ ] `fhetch_integration_guide.md`
- [ ] Hardware integrator white paper
- [ ] API reference document
- [ ] HEIR and FHETCH contract specification

---

## Phase 9: Integration Testing & Validation

### 9.1 Cross-Scheme Validation

**Test that the supported schemes (CKKS, BGV) lower correctly.** CGGI does not
route through `polynomial` and is out of scope (§3.2 Step 4).

```bash
# Test CKKS path
heir-opt tests/Examples/common/dot_product_8f.mlir \
  --mlir-to-ckks --scheme-to-fhetch=scheme=ckks target=generic \
  -o "$HOME/tmp/ckks_fhetch.mlir"

# Test BGV path
heir-opt tests/Examples/common/matmul.mlir \
  --mlir-to-bgv --scheme-to-fhetch=scheme=bgv target=generic \
  -o "$HOME/tmp/bgv_fhetch.mlir"
```

### 9.2 Parameter Space Testing

**Validate a matrix of parameter combinations:**

| Degree | RNS Limbs | Scheme | Test Status |
|--------|-----------|--------|-------------|
| 2048 | 3 | CKKS | ✓ |
| 2048 | 4 | CKKS | ✓ |
| 4096 | 4 | CKKS | ✓ |
| 8192 | 5 | CKKS | ✓ |
| 2048 | 2 | BGV | ✓ |
| 4096 | 3 | BGV | ✓ |
| ~~1024~~ | ~~N/A~~ | ~~CGGI~~ | out of scope (§3.2) |

### 9.3 Performance Profiling

**Measure compilation overhead & code quality:**

```cpp
// Pseudo-benchmark
benchmark::Run("scheme_to_fhetch_pipeline", [](benchmark::State& state) {
  for (auto _ : state) {
    // Load MLIR
    auto module = loadMLIR("dot_product_8f.mlir");

    // Run pipeline
    auto context = createContext();
    auto pm = PassManager(context);
    addSchemeToFhetchPipeline(pm, opts);
    pm.run(module);

    // Measure: compile time, IR size growth, pass time breakdown
  }
});
```

### 9.4 Deliverables

- [ ] Cross-scheme validation test suite
- [ ] Parameter space coverage matrix
- [ ] Compile-time performance benchmarks
- [ ] Generated IR size analysis

---

## Phase 10: Project Management & Handoff

### 10.1 Milestone Breakdown

**Updated timeline leveraging Niobium compiler:**

| Phase | Milestone | Timeline | Scope | Notes |
|-------|-----------|----------|-------|-------|
| 0 | **Upstream prerequisites** | **Week 1–4** | **Compiler** | **CKKS `rotate`/`rescale` lowering (§0.2); the `lwe-to-polynomial` fix (§0.4) as a separate track. Gates only Phases 6–7; Phases 2–5 are tested from hand-written `polynomial` and `fhetch` IR.** |
| 1 | Spec analysis & requirements | Week 1 | Requirements | Align with Niobium IR spec; runs in parallel with Phase 0 |
| 2 | Fhetch dialect design & tests | Week 4-5 | Dialect | Reduced: reuses `!polynomial.polynomial` (§2.1) |
| 3 | Lowering passes | Week 6-8 | Compiler | `polynomial-to-fhetch` only (§3.1) |
| 4 | Optimization passes | Week 9-10 | Optimization | Rotation order, resource allocation; NTT scheduling and CSE dropped (§4.4) |
| 5 | Code generation | Week 11 | Backend | Emit FHETCH IR for Niobium |
| 6 | Pipeline & CLI integration | Week 12 | Integration | No change from original |
| 7 | E2E testing framework | Week 13 | QA | **Reduced complexity:** no ref interpreter needed |
| 8 | Validation & performance | Week 14-15 | QA/Performance | Test with real Niobium compiler |
| 9 | Documentation | Week 16 | Documentation | API guide, Niobium integration notes |
| 10 | Completion | Week 17 | Project | **Total: 17 weeks** |

The original 15-week estimate assumed all of Phases 2–4 were greenfield. The
audit moves work in both directions: Phase 0 adds ~3 weeks of gating upstream
work, while Phases 2 and 4 shrink because the NTT scheduling pass, the bespoke
CSE, and the parallel polynomial type are no longer being built (§0.3). Net +2
weeks, with materially less code to maintain.

**Key simplifications from Niobium integration:**
- Simpler test harness (just call Niobium-compiled functions)
- FHETCH IR validation is focused on format/structure, not correctness simulation

**Key simplifications from the HEIR audit:**
- No `fhetch` polynomial type, NTT-domain attribute, or modulus attribute (§2.1)
- No NTT scheduling pass, no `fhetch`-specific CSE (§4.2.1, §4.2.4)
- Scheme semantics resolved before `fhetch` is reached, so no per-scheme lowering
  logic in this project's code (§3.2)

### 10.2 Deliverables Checklist

**Prerequisites (upstream HEIR, not `fhetch` code):**
- [ ] Fix the `lwe-to-polynomial` plaintext type mismatch + 3 lit tests (§0.4) — separate track
- [ ] `ckks.rotate` → Galois + `lwe.key_switch_inner` (§0.2, upstream #2886)
- [ ] `ckks.rescale` → `lwe.mod_down` (§0.2)

**Code:**
- [x] `lib/Dialect/Fhetch/` (dialect definition)
- [x] `lib/Dialect/Polynomial/Conversions/PolynomialToFhetch/` (lowering)
- [ ] `lib/Dialect/Fhetch/Transforms/` (4 optimization passes — 2 of the original
      6 were dropped as duplicates, §4.4)
- [ ] `lib/Target/Fhetch/` (codegen)
- [ ] `lib/Pipelines/ArithmeticPipelineRegistration.cpp` (pipeline)
- [x] `tests/Dialect/Fhetch/` (dialect IR tests)
- [ ] `tests/Examples/fhetch/` (E2E tests)

**Documentation:**
- [ ] `docs/content/en/docs/Guides/fhetch_integration_guide.md`
- [ ] `docs/content/en/docs/Dialects/fhetch.md` (operation reference)
- [ ] Hardware integrator guide (white paper)
- [ ] Design document: HEIR and FHETCH contract

**Tests:**
- [x] Dialect syntax tests (lit/FileCheck, no `CHECK-LABEL`)
- [x] Conversion pass tests (IR matching)
- [ ] FHETCH IR validation (format & semantics checks)
- [ ] E2E tests via Niobium compiler (CKKS + BGV; CGGI out of scope, §3.2)
- [ ] Niobium compiler integration tests
- [ ] Performance benchmarks (compile time, generated code quality)

### 10.3 Acceptance Criteria

**Definition of Done:**

1. **Functional:**
   - [ ] All CKKS/BGV operations lower to fhetch dialect (CGGI out of scope, §3.2)
   - [ ] All optimization passes run without errors
   - [ ] Generated FHETCH IR validates against specification
   - [ ] E2E tests pass for ≥5 parameter combinations per scheme

2. **Performance:**
   - [ ] Compile-time overhead < 20% vs. baseline HEIR pipeline
   - [ ] Generated code size within 10% of hand-written reference

3. **Integration:**
   - [ ] FHETCH specification consensus with consortium members
   - [ ] Niobium (at minimum) validates IR on hardware
   - [ ] Documentation sufficient for 3rd-party hardware vendors

4. **Testing:**
   - [ ] Code coverage ≥ 80% for new passes
   - [ ] All tests pass in CI/CD

### 10.4 Risk Mitigation

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **CKKS `rotate`/`rescale` never reach `polynomial` (§0.2)** | **High — it is the state of `main` today** | **Blocking** | Implement the lowering ourselves and upstream it against #2886; treat as a gating prerequisite, not a parallel track |
| FHETCH spec changes mid-integration | Medium | High | Establish spec freeze date; version IR |
| Hardware simulator unavailable | Medium | Medium | Build reference interpreter early |
| RNS parameter incompatibilities | Medium | Medium | Design flexible parameter negotiation |
| `fhetch` drifts into duplicating `polynomial` (§0.3) | Medium | High | Require written justification per op (§2.1); review against the reuse table |
| Performance regression | Low | High | Establish perf baselines in Week 2 |

### 10.5 Deliverables

- [ ] Project timeline & milestone chart
- [ ] Detailed task breakdown (GitHub issues)
- [ ] Risk register & mitigation strategies
- [ ] Handoff documentation (how to maintain the code)

---

## Appendix: Reference Resources

### A.1 HEIR Documentation
- https://heir.dev/
- https://github.com/google/heir/
- Backend Integration Guide: https://heir.dev/docs/guides/new_backend/
- MLIR Tutorials (Jeremy Kun): https://github.com/j2kun/mlir-tutorial

### A.2 FHETCH Consortium
- https://fhetch.org/
- FHETCH Polynomial IR Draft Specification
- Contact: inquiries@fhetch.org
- Founding members: Niobium Microsystems, Optalysys, Chain Reaction

### A.3 MLIR References
- MLIR Dialects: https://mlir.llvm.org/docs/Dialects/
- Dialect Conversion: https://mlir.llvm.org/docs/DialectConversion/
- Pass Infrastructure: https://mlir.llvm.org/docs/PassManagement/

### A.4 FHE Schemes
- CKKS (Cheon et al.): https://arxiv.org/abs/2101.07055
- BGV (Brakerski et al.): https://eprint.iacr.org/2011/277
- CGGI (Chillotti et al.): https://eprint.iacr.org/2016/770

### A.5 Hardware Acceleration Papers
- "Towards a Polynomial Instruction Based Compiler for Fully Homomorphic Encryption": https://eprint.iacr.org/2024/707

---

## Appendix: Example Code Snippets

### B.1 FHETCH Dialect Definition (Tablegen Snippet)

Operand types are TableGen constraint defs, not MLIR type syntax — `!fhetch.polynomial:$lhs`
does not parse. The implemented dialect defines a single-residue constraint on
`!polynomial.polynomial`, with form-specific variants, and reuses
`SameOperandsAndResultForm` from the `polynomial` dialect for the domain rule.
Excerpt of `lib/Dialect/Fhetch/IR/FhetchOps.td`:

```tablegen
class Fhetch_PolynomialBase<list<Pred> preds, string desc> : Type<
    And<[Polynomial_PolynomialType.predicate,
         CPred<"::llvm::isa<::mlir::heir::mod_arith::ModArithType>("
               "::llvm::cast<::mlir::heir::polynomial::PolynomialType>($_self)"
               ".getRing().getCoefficientType())">,
         CPred<"::llvm::cast<::mlir::heir::polynomial::PolynomialType>($_self)"
               ".getRing().getPolynomialModulus() != nullptr">] # preds>,
    desc, "::mlir::heir::polynomial::PolynomialType">;

def Fhetch_Polynomial : Fhetch_PolynomialBase<[],
    "single-residue polynomial">;
def Fhetch_CoeffPolynomial : Fhetch_PolynomialBase<[Fhetch_FormPred<"COEFF">],
    "single-residue polynomial in coefficient form">;
def Fhetch_EvalPolynomial : Fhetch_PolynomialBase<[Fhetch_FormPred<"EVAL">],
    "single-residue polynomial in evaluation form">;

class Fhetch_BinaryOp<string mnemonic, list<Trait> traits = []> :
        Fhetch_Op<mnemonic,
                  traits # [SameOperandsAndResultType, SameOperandsAndResultForm]> {
  let arguments = (ins Fhetch_Polynomial:$lhs, Fhetch_Polynomial:$rhs);
  let results = (outs Fhetch_Polynomial:$result);
  let assemblyFormat = "operands attr-dict `:` type($result)";
}

def Fhetch_MulOp : Fhetch_BinaryOp<"mul", [Commutative]> { ... }

def Fhetch_NTTOp : Fhetch_Op<"ntt", [
    TypesMatchWith<"output is the input ring in evaluation form",
                   "input", "output", Fhetch_InForm<"EVAL">.result>]> {
  ...
  let arguments = (ins Fhetch_CoeffPolynomial:$input);
  let results = (outs Fhetch_EvalPolynomial:$output);
}
```

`Fhetch_Op` adds `Pure` (rather than bare `NoMemoryEffect`) to every op, which is
what lets generic `--cse` and DCE handle them, per §4.2.4, and `hasFolder`.

### B.2 Lowering Pattern Example

Two corrections from the original draft: the pass lowers from `polynomial`, not
`ckks` (§3.1), and op construction uses `OpType::create(rewriter, ...)` —
`rewriter.create<OpType>(...)` is deprecated in HEIR (`AGENTS.md`).

The arithmetic patterns are the generic `ConvertLimbwise` shown in §3.2 Step 3.
The NTT domain is a property of the type, not a separate analysis: no
`isInNTTDomain()` helper and no on-the-fly NTT insertion is needed, and the
coefficient-form check runs once, before the conversion (§3.2 Step 1). The
patterns that only move limbs are smaller still; from
`lib/Dialect/Polynomial/Conversions/PolynomialToFhetch/PolynomialToFhetch.cpp`:

```cpp
struct ConvertExtractResidue
    : public OpConversionPattern<rns::ExtractResidueOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      rns::ExtractResidueOp op, OneToNOpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    ValueRange limbs = adaptor.getInput();
    size_t index = op.getIndex().getZExtValue();
    if (index >= limbs.size())
      return rewriter.notifyMatchFailure(op, "index out of range");
    rewriter.replaceOpWithMultiple(op, {limbs.slice(index, 1)});
    return success();
  }
};
```

---

