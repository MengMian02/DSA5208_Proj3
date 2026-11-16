# WORKSPACE_SEPARATION

Date: 2026-11-15. The original working directory
`DSA5208_Proj3` was not deleted. It was fully backed up, then split into two
independent directories.

## Backup

* Backup: `/Users/hc/Desktop/NUS DSML/DSA 5208/DSA5208_Proj3_Backup_2026-11-15`
* Size: 103 MB, 610 files including `.git`
* Verified: `git rev-parse HEAD` = `f3bdca3`; branch `repair-2026-11-14`;
  clean status; `rsync -an --delete` against the source produced no differences.

## Directory comparison

| Property | `DSA5208_Proj3_GitHub` | `DSA5208_Proj3_Harry` |
|---|---|---|
| Path | `/Users/hc/Desktop/NUS DSML/DSA 5208/DSA5208_Proj3_GitHub` | `/Users/hc/Desktop/NUS DSML/DSA 5208/DSA5208_Proj3_Harry` |
| Git mode | detached worktree | normal branch |
| Branch | detached HEAD | `repair-2026-11-14` |
| HEAD SHA | `c450025` = `origin/main` | `f3bdca3` (merge of origin/main and personal work) |
| Remote | `https://github.com/MengMian02/DSA5208_Proj3.git` | same remote, never pushed |
| Tracked worktree | clean (`git diff --stat` empty) | clean |
| Results directory | `results/` (GitHub/teammate results) | `results/` plus `results/vanda_harry/` |
| Results files | 50 | 69 |
| Non-report source files | 147 | 218 |
| Personal `repair-2026-11-14` code | absent | present |
| Fixed-total resampling | absent (as in origin/main) | present |
| E0–E9 Vanda kit | absent | present |
| Personal HPC job records | absent | present (`1557208`, `1557218`–`1557223` in the manifest) |
| Report artifacts | `report/`, `docs/RESULT_AUDIT_GITHUB.md` (untracked additions) | personal docs/results only |

## Why the baseline is a worktree

`DSA5208_Proj3_GitHub` was created with
`git worktree add --detach DSA5208_Proj3_GitHub c450025`. It therefore checks
out exactly the commit that `origin/main` points to, without copying personal
branch files. Its tracked tree has no diff from `c450025`. The only untracked
additions are the new report/audit material produced for this submission.

## Personal provenance preserved in Harry

The Harry directory retains the `repair-2026-11-14` history, personal source
modifications, E0–E9 scripts, Vanda PBS kit, `results/vanda_harry/`, submission
manifest and the historical personal Job IDs. None of these is used as
evidence in the first report, and none was pushed to GitHub.

## Safety notes

* No `git reset --hard`, `git clean -fd`, or project deletion was used.
* No Vanda job was submitted, canceled or queried in this separation step.
* No file was pushed to the remote.
* The GitHub baseline was not modified in its tracked files.
