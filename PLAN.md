# Finish lowering-time fusion

- [x] Read #69, #166, the preceding layers and the worktree rules. Record the start on #69.
- [x] Pin the folded seed layout and the interior-point KKT assembly against NumPy. Main matches the values and fails the storage and loop gates.
- [x] Trace the neural MPC declarations after the tensor reproduction. The folded derivatives write through direct layout maps. Active seed planes stay compact between products with matching domains.
- [x] Fuse KKT assembly into one lowered loop. The fixture matches NumPy and has no private array.
- [ ] Record the fail-before and perturbation runs for the new gates.
- [ ] Regenerate and audit the snapshots. Record source, generation, C compilation, and workspace measurements against the #503 base. Count neural MPC caller workspace and root local arrays for both horizons at each requested lane setting.
- [ ] Run the detached pre-merge checks. Open the draft, complete the Opus review and fix check, and hand off on #69.
