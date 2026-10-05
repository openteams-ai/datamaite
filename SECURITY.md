# Security Policy

This document describes how to report vulnerabilities in datamaite and how the
maintainers handle findings from automated security scanners.

## Reporting a vulnerability

Please do **not** report security vulnerabilities in an ordinary or public
issue. Report them privately through either channel:

- **JATIC users:** open a
  [new issue](https://gitlab.jatic.net/jatic/orchestration-interoperability/datamaite/-/issues/new)
  on the JATIC GitLab and select **This issue is confidential** before creating
  it, so only project members can see it.
- **Everyone else:** use GitHub's
  [private vulnerability reporting](https://github.com/openteams-ai/datamaite/security/advisories/new),
  which is visible only to the maintainers.

Include:

- A description of the issue and its impact
- Steps to reproduce (a minimal script, dataset layout, or `datamaite` CLI
  command is ideal)
- Affected versions or commits
- Your name and (optionally) a credit preference

For non-security bugs, follow the regular process in
[CONTRIBUTING.md](CONTRIBUTING.md).

## Supported versions

Only the latest minor release receives security fixes.

| Version        | Status      | Security fixes |
| -------------- | ----------- | -------------- |
| `0.5.x`        | active      | yes            |
| anything older | unsupported | no             |

## Automated scanner coverage

The CI pipeline runs these security checks (see
[.gitlab-ci.yml](.gitlab-ci.yml) and
[.pre-commit-config.yaml](.pre-commit-config.yaml)):

| Scanner | Scope | Suppression mechanism |
| --- | --- | --- |
| Bandit and ruff `S` rules (pre-commit) | Python source | per-line `# nosec <test-id>` or `# noqa: S<code>` with a justification |
| SAST (JATIC `dr-compliance` component) | source code, excluding `tests` and `tools` | GitLab Vulnerability Report dismissal (with comment) |
| Dependency scanning (JATIC `dr-compliance` component) | `requirements.txt`, generated from `uv.lock` | GitLab Vulnerability Report dismissal (with comment) |
| GitLab Secret Detection | commits in MR and `main` pipelines | do not commit secrets |

Findings flow into the GitLab Vulnerability Report.

## Dismissing a false positive

The same workflow applies whichever scanner raised the finding.

1. **Verify it is a false positive.** Read the rule or CVE description, inspect
   the code or dependency, and confirm the finding does not apply to how
   datamaite uses the affected component. If you are unsure, treat it as a
   true positive and fix it.

2. **Get a second opinion.** Security suppressions land through a normal merge
   request reviewed by a maintainer. The merge request description must link to
   the finding (rule ID or CVE) and to any upstream advisory or discussion that
   supports the dismissal.

3. **Record the suppression in the right place.** The justification must be
   specific to this project. Generic statements like "not exploitable in our
   case" will be rejected in review.
    - **Bandit or ruff:** add `# nosec <test-id>` or `# noqa: S<code>` on the
      triggering line, with the reason in the same comment.
    - **Dependency scanning:** prefer bumping the affected package and
      committing the updated `uv.lock` and `requirements.txt`. When no fix is
      available upstream, dismiss the finding in the GitLab Vulnerability
      Report with a comment containing:
        1. The CVE ID
        2. Why datamaite is not exploitable (for example, the affected API is
           never called)
        3. A re-evaluation date
    - **SAST:** dismiss the finding in the GitLab Vulnerability Report with the
      same three-part comment.
    - **Secret Detection:** a true positive must be rotated immediately, because
      the secret is already in git history and must be assumed compromised.

4. **Set an expiry.** Every dismissal carries a re-evaluation date (90 days for
   HIGH or CRITICAL, 180 days for MEDIUM). When the date passes, the dismissal
   must be removed or re-justified.

## Hardcoded-secret policy

The project ships no hardcoded secrets. The `datamaite-e2e-secret` values in the
S3 end-to-end CI job are throwaway credentials for that job's ephemeral
S3-compatible service, not real secrets. If you discover what looks like a real
credential, password, token, or private key in this repository, even in a test
fixture, report it through the confidential channel above rather than an
ordinary issue.
