# Backlog — voice device

Decided work not yet done. Distinct from `specs-audit/findings-voice.md`, which is a survey of what the
code currently does.

---

## Raise the default wake-word threshold to 0.99

**Operator decision.** The openWakeWord activation threshold ships at `0.90`; make it `0.99`.

Three places currently disagree, so all of them move together — the mismatch is
`specs-audit/findings-voice.md` F28:

| where | now | to |
|---|---|---|
| `wake_voice_service.py:60` — `DEFAULT_OPENWAKEWORD_THRESHOLD` | `0.90` | `0.99` |
| `wake_voice_service.py:424` — the `--openwakeword-threshold` help text | says "or **0.85**" | `0.99` |
| `.env.example:35` — `# VOICE_OPENWAKEWORD_THRESHOLD=` | `0.90` | `0.99` |

The env var and the CLI flag already override it, so nothing else needs to change.

**Why it matters more than a tuning tweak.** A false wake is not merely an annoyance on this device — per
findings F1 and F2 it is the entry point to two real problems:

- F1 — in relay mode (the default) the "nobody spoke" timeout measures from `_last_activity`, which every
  inbound frame bumps, so a false wake in a room with a television can stream the room for up to the 300 s
  hard cap rather than stopping at 20 s.
- F2 — every exchange's transcript is printed to stdout, and `docker-compose.yml` sets no logging limits, so
  each false wake leaves a durable on-disk record of whatever the room was saying.

Raising the threshold reduces how often that path is entered. It does **not** fix either finding, and
should not be treated as having done so.

**Worth checking after the change:** 0.99 is a strict threshold for openWakeWord, so confirm the wake word
still fires reliably for quiet or distant speech before considering this settled — the failure mode of
setting it too high is a device that appears deaf, which is harder to diagnose than a false wake.
