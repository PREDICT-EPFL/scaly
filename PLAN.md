# Finish lowering-time fusion

- [x] Read #69, #166, the preceding layers and the worktree rules. Record the start on #69.
- [x] Pin the folded seed layout and the interior-point KKT assembly against NumPy. Main matches the values and fails the storage and loop gates.
- [ ] Trace the neural MPC declarations after the tensor reproduction. Remove affine copies using the lowering access maps. Keep storage for whole-buffer consumers and unchanged reduction order.
- [ ] Fuse KKT assembly into one lowered loop. Perturb each new gate and restore it before validation.
- [ ] Regenerate and audit the snapshots. Record source, generation, C compilation, and workspace measurements against the #503 base. Count neural MPC caller workspace and root local arrays for both horizons at each requested lane setting.
- [ ] Run the detached pre-merge checks. Open the draft, complete the Opus review and fix check, and hand off on #69.
