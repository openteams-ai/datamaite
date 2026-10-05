# Contributing to datamaite

datamaite is developed by the JATIC Orchestration and Interoperability team on
the [JATIC GitLab](https://gitlab.jatic.net/jatic/orchestration-interoperability/datamaite).
This page covers how to report problems, how changes are made, and the branching
model the project follows.

## Who can contribute

- **JATIC program members** with access to the project can contribute code and
  documentation through merge requests, following the workflow below.
- **Everyone else** is welcome to report bugs and request features (see below).
  We do not currently accept code contributions from outside the JATIC program.

## Reporting bugs and requesting features

Bugs and feature requests are tracked as issues in either of two places, and
the maintainers triage both:

- **JATIC users** with a JATIC GitLab account:
  [GitLab Issues](https://gitlab.jatic.net/jatic/orchestration-interoperability/datamaite/-/issues)
  on this project.
- **Everyone else:** the public
  [GitHub issue tracker](https://github.com/openteams-ai/datamaite/issues).

Issues are the record of reports, discussion, and responses, wherever they were
filed.

**Do not report security vulnerabilities as ordinary issues.** Follow
[SECURITY.md](SECURITY.md) instead.

For a **bug**, please include:

- the datamaite version (`python -c "import datamaite; print(datamaite.__version__)"`),
  Python version, and operating system;
- the dataset format and the call or `datamaite` CLI command you ran;
- what you expected and what happened, with the full traceback or validator
  output;
- a minimal dataset or layout that reproduces it, if you can share one.

For a **feature request**, describe the task you are trying to do, the dataset
formats involved, and what you would like datamaite to do.

## Making a change

1. **Start from an issue.** All work is tracked in GitLab Issues; open one first
   if none exists.
2. **Branch from `main`** using the issue number as a prefix:
   `<issue>-<short-description>` (for example `120-markdown-lint-link-check`).
3. **Set up and check locally** with the workflow in
   [README_DEV.md](README_DEV.md): `uv sync --all-extras`, `uv run pytest`,
   `uv run pre-commit run --all-files`, and `uv run pyright src/`. If you
   change a dependency, commit the updated `uv.lock` and `requirements.txt`
   too.
4. **Add a changelog entry** under `## [Unreleased]` in
   [CHANGELOG.md](CHANGELOG.md), citing the issue number.
5. **Open a merge request** into `main`. Reference the issue in the title, for
   example `Add markdownlint to CI (#120)`, and put `Closes #<issue>` in the
   description.
6. **Review and merge.** A merge request needs one approval and a passing
   pipeline. Only maintainers can merge to `main`.

## Branching strategy

datamaite follows **GitHub Flow**:

- `main` is protected and always releasable. Nobody pushes to it directly.
- Every change is made on a short-lived branch named after its issue and lands
  through a merge request.
- Releases are cut from `main` by pushing a version tag (`X.Y.Z`). There are no
  release or develop branches. The tag-driven release process is described in
  [README_DEV.md](README_DEV.md#releasing-85).

Deviations from stock GitHub Flow:

- Merge requests are **squashed and fast-forward merged**, so `main` has linear
  history with one commit per merge request. Rebase your branch on `main` before
  merging if GitLab asks.
- Releases are marked by tags on `main` rather than by deploying each merge.
