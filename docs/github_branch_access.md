# Diagnosing "Tworzenie gałęzi nie jest autoryzowane" Errors

When GitHub displays the Polish message **"Tworzenie gałęzi nie jest autoryzowane dla tego repozytorium. Sprawdź swoje uprawnienia GitHub."** it is refusing to create a branch on the remote repository. This happens before any pull request can be opened and usually points to repository permissions or policies.

## Common causes

1. **Insufficient permissions** – Your GitHub account (or the token used by an automation) lacks *Write* access. View access is not enough to create a branch.
2. **Branch creation disabled** – Organization administrators can forbid branch creation outside of specific workflows or require forks instead of direct branches.
3. **Missing personal access token scopes** – If you authenticate over HTTPS, ensure the token includes the `repo` scope. For SSH, confirm the associated key is registered with the correct account.
4. **Pending SSO authorization** – Enterprise organizations that require SSO need each personal access token or SSH key to be authorized for that organization.
5. **Repository rulesets/branch protection** – New GitHub rulesets can restrict branch patterns. Creating a branch that matches a protected pattern without approval triggers this error.

## How to investigate

1. Run `git remote -v` to confirm you are pushing to the intended repository.
2. Execute `git ls-remote --heads origin` to verify that authentication succeeds; failures indicate missing permissions or token scopes.
3. Ask a repository administrator to check **Settings → Manage access** and confirm your role is at least **Write**.
4. Review organization policies under **Settings → Policies → Repository rules** (or request an admin to do so) to see whether branch creation is limited to certain users or patterns.
5. If you rely on a personal access token, visit **https://github.com/settings/tokens** to confirm it is active and has the `repo` scope; re-authorize it for the organization if SSO is enabled.
6. For SSH authentication, run `ssh -T git@github.com` to ensure the key is recognized by GitHub and mapped to the correct user.

### When using the Codex workspace

- The container does **not** ship with any GitHub token. Verify the token you paste into the environment (for example via `git config credential.helper store` or `git credential-cache`) grants **Contents: read & write** (fine-grained) or the classic `repo` scope; read-only tokens trigger the branch-creation error even if you own the repository.
- If you rotated credentials recently, clear any cached entries (`git credential-cache exit` or removing `~/.git-credentials`) so the workspace stops presenting an outdated token.
- Confirm that the token is associated with the same GitHub account that owns the repository. Using a token from a different account that only has public-read access will also produce the authorization failure.

## Workarounds while waiting for access

- **Fork the repository** under an account you control, push changes to the fork, and open a pull request from the fork.
- **Request a maintainer** to create a feature branch on your behalf and grant you temporary access to that branch.

Once your account (or token) has sufficient permissions and complies with any organization policies, pushing a branch from this environment and opening a pull request will succeed.
