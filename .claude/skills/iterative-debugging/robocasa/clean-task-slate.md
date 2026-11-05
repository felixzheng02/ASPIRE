# Clean Task Slate: RoboCasa

Use this before rerunning OpenDrawer.

1. Confirm the campaign root, suite (`OpenDrawer`), and whether `$TASK_DIR` should be archived.
   The task is self-contained in `$RUN_ROOT/robocasa/OpenDrawer/task_0/`. Moving that folder
   aside is a complete clean slate for the task. To set it aside, `mv` it to an `archive/`
   directory; never `rm`.
2. Stop the **simulator** by the PID in `$TASK_DIR/simulator.pid` only. Never `pkill`, never a
   pattern match. If the PID file is missing, look up the process from `simulator.log` and kill
   that PID.
3. **Never kill SAM3.** `GET /healthz` on `http://127.0.0.1:8114` must stay 200 across a slate
   clean. If it is down, from `$WORKSPACE_ROOT` run `./scripts/ensure_sam3.sh` — start-if-idle,
   not a restart.
4. Confirm no `cap-harness run` / `run-batch` remains (`pgrep -af 'cap-harness' | grep -v pgrep`).
   Kill leftover harness PIDs by PID, not by pattern.
5. Keep the three seed bands in `campaign.json` separate: development 101 102 103, unseen-check
   126, held-out 1 2 3. A rerun that debugs against 126 has spent the only cheap evidence that
   the program generalizes; archive the task folder and start clean.
6. Do not read external baseline code or outputs. Do not probe 8115/8116/8118.
7. Workers write task artifacts and findings; this campaign has no shared skill library to edit.
