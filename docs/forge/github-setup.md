# GitHub Setup

## Create The Repo

1. Create a private GitHub repository named `forge`.
2. Do not initialize it with a README, license, or `.gitignore`; this scaffold
   already has those files.
3. Push the local scaffold to `main`.

Suggested commands from a clean copy:

```bash
git init
git add .
git commit -m "Initial Forge scaffold"
git branch -M main
git remote add origin git@github.com:<org-or-user>/forge.git
git push -u origin main
```

## Add Team Members

For a personal repo:

1. Go to `Settings -> Collaborators`.
2. Add Devansh, Xhitij, and Sasank by GitHub username.
3. Give `Write` access.

For an organization repo:

1. Create a `forge-core` team.
2. Add Devansh, Xhitij, and Sasank.
3. Give the team `Write` access to the repo.

## Branch Protection

Enable branch protection on `main`:

- Require pull requests before merging.
- Require at least one approval.
- Require conversation resolution before merge.
- Require status checks once CI is active.
- Block force pushes.
- Block direct pushes to `main`.

## First Sprint Branches

Use short, searchable branch names:

```text
kernel/swiglu-devansh
kernel/rope-xhitij
kernel/cross-entropy-devansh
infra/repo-design-sasank
```
