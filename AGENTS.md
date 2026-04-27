# FastSIMUS

Array API-compliant ultrasound simulation library with NumPy/JAX/CuPy backends for 50-100x GPU acceleration.

## Code Organization

- src/ layout, organize by algorithm/domain
- Many small files over few large files
- 200-400 lines typical, 800 max per file
- No emojis in code, comments, or documentation
- `uv` package manager

## Key Constraints

- ALL numerical operations via Array API (`xp` namespace) -- see `array-api-typing` skill
- Never mutate input arrays (breaks JAX)
- Validate against PyMUST reference implementation (rtol=1e-4)
- TDD: write tests first -- see `python-unit-testing` skill

## File Structure

```
src/fast_simus/
  __init__.py
  transducer_params.py   # Transducer parameter model
  transducer_presets.py   # Preset transducer configs (P4-2v, L11-5v, etc.)
  medium_params.py        # Medium/tissue parameters
  tx_delay.py             # Transmit delay computation
  spectrum.py             # Pulse & probe spectrum functions
  pfield.py              # RMS pressure field computation
  utils/
    __init__.py
    _array_api.py         # "Array" type for Array API
    geometry.py           # Geometry utilities
tests/
  conftest.py
  test_transducers.py
  test_tx_delay.py
  test_spectrum.py
  test_pfield.py
  utils/
    test_array_api.py
```

## Workflow

- Conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`
- All tests must pass: `poe test`
- Linting and type-checking: `poe lint`

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
