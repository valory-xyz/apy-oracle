
# Release Process from develop to main

1. Make sure all tests pass, coverage is at 100% and the local branch is in a clean state (nothing to commit). Make sure you have a clean develop virtual environment.
   
2. Determine the next `APY` version. Create new release branch named `feature/release-{new-version}`, switch to this branch. Update the version in all the packages. Update all hashes and auxiliary info with `tox generators`. Commit if satisfied.

3. Check the package upgrades are correct by running `autonomy check-packages`. Commit if satisfied.

4. Write release notes and place them in `HISTORY.md`. Add upgrading tips in `upgrading.md`. If necessary, adjust version references in `SECURITY.md`. Commit if satisfied.

5. Open PRs and merge into develop. Then open develop to main PR and merge it.

6. Tag version on main.

7. Pull main. Release packages into registry: `autonomy init --reset --author valory --ipfs --remote` and `autonomy push-all`. If necessary, run it several times until all packages are updated.

8. Build and tag images for the documentation. `VERSION=TAG-TO-BE-RELEASED make release-images`. Inform DevOps of new release so that these images can be rolled out.
