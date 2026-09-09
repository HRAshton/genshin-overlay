# Genshin Overlay

Screen-only Windows HUD reader and transparent cooldown overlay. It finds the
rendered game viewport, locates E/Q controls, identifies the active party slot,
and keeps observed cooldowns alive after character switches.

## Install and run

```powershell
python -m pip install -r requirements.txt
python main.py --debug
```

Use Genshin borderless or windowed mode. Debug mode opens a detector window.
Run without `--debug` after calibration:

```powershell
python main.py
```

Close from console with `Ctrl+C`. Static detector proof:

```powershell
python tools/inspect_references.py
python -m unittest discover -s tests -v
```

Static debug images are written to `debug_output/`.

## Capturing detection failures

Debug mode automatically writes a diagnostic bundle whenever a new cooldown is
confirmed. Use **Save diagnostic bundle** in the debug window to capture the
current state manually. Bundles are stored under `debug_sessions/` and contain:

- the current full-resolution frame and annotated frame;
- the preceding two seconds of native-resolution E/Q crops and masks;
- accepted and rejected glyph boxes with best and second-best template scores;
- HUD, party portrait, reading, confidence, tracker decisions, and emitted
  overlay-item metadata.

`frame.png` can contain private desktop content. Review each bundle before sharing
it. Diagnostic sessions are excluded from version control by default.

Templates for `2`, `5`, `7`, and `8` come from supplied game fixture. Remaining
digits use conservative Windows-font fallbacks. Replace fallbacks with game-font
samples during live calibration for best confidence.
