# Datamaite Documentation

`datamaite` is a unified framework for working with image classification, object
detection, and full motion video datasets.

It exists to do three things: provide MAITE-compliant dataset wrappers for
loading and writing datasets, perform conversions between dataset formats, and
run HMIE validation on datasets before you rely on them.

At its core, `datamaite` loads each input format into a neutral,
source-preserving in-memory model. The box-track model, BoxTrackDataset, doesn't
just describe a dataset — it is a MAITE multi-object-tracking dataset.

MAITE (Modular AI Trustworthy Engineering), developed under the JATIC program,
is a library of common protocols — structural subtypes — that let datasets and
models interoperate across T&E tooling without tight coupling. Because MAITE
compliance is structural, `datamaite`'s objects satisfy these interfaces
directly - so any MAITE-aware evaluation harness can consume a `datamaite`
dataset as-is. These `datamaite` objects are concrete implementations of the
corresponding MAITE protocols.

::::{grid} 1 1 1 3
:gutter: 3

:::{grid-item-card} {octicon}`rocket` Getting Started
:link: getting-started/index
:link-type: doc

Install datamaite and run your first dataset load, conversion, and validation.
:::

:::{grid-item-card} {octicon}`book` Tutorials
:link: tutorials/index
:link-type: doc

Hands-on, runnable notebooks for loading, converting, and validating datasets.
:::

:::{grid-item-card} {octicon}`tools` Reference
:link: reference/index
:link-type: doc

The codebase architecture and the `datamaite` command-line interface.
:::

::::

## About the JATIC program and CDAO

The Joint AI Test Infrastructure Capability (JATIC) program develops software
products for AI Test & Evaluation (T&E) and AI Assurance. The program is managed
by the Assessment & Assurance Division of the
DoD Chief Digital and Artificial Intelligence Office (CDAO).
It is funded from FY23-FY29.

:::{admonition} Program Mission

Develop software to accelerate and enable AI model test and evaluation for
testers across the Department of Defense (DoD) enterprise, including DoD
programs, research laboratories, industry partners, and academia in order to
provide insight on the performance, effectiveness, robustness, and safety of the
DoD's AI-enabled systems.
:::

Learn more in the
[CDAO JATIC program documentation](https://cdao.pages.jatic.net/public/).

### CDAO Funding Acknowledgment

```{include} ../README.md
:start-after: <!-- --8<-- [start:acknowledgment] -->
:end-before: <!-- --8<-- [end:acknowledgment] -->
```

```{toctree}
:maxdepth: 2
:hidden:

getting-started/index
tutorials/index
reference/index
```
