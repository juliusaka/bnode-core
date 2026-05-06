# Trainer workflow feedback request

Please review `trainer_flow.md` before any refactor or restart-workflow implementation.

The current intent is:

1. use `trainer_flow.md` as the source-of-truth map of the current control flow
2. agree on better function boundaries before touching restart logic
3. only then implement the cleanup in `trainer.py`

## Please comment on

1. which helper-function boundaries you want first
2. which variables should be treated as true saved state
3. which variables should stay local and never be checkpointed
4. whether the proposed outer-state vs inner-state split matches the restart plan

## Feedback

Add notes here before implementation.
