## LMM Formula

**Response variable:**
- `response`: mean firing rate (baseline-subtracted) in 0-100 ms post-trigger window

**Fixed effects:**
- `condition`: within-neuron factor. `Track` (animal controls frequency) vs `Playback` (frequency is replayed)
- `mapping`: between-session factor. `MP1` (playback sessions, no mapping change) vs `MP2` (mapping_change_only sessions, mapping change occurred)
- `expertise`: between-session factor. `Beginner` (first half of sessions) vs `Expert` (second half of sessions)
- `half`: within-session factor. `H1` (first half of trial, 0.0-0.5 s) vs `H2` (second half of trial, 0.5-1.0 s)

**Random effects:**
- `neuron_id`: random intercept — each neuron has its own baseline firing rate
- `condition | neuron_id`: random slope — each neuron has its own Track-to-Playback modulation magnitude

**Full model formula (R / statsmodels syntax):**
```
response ~ condition * mapping * expertise * half
```

This expands to all main effects plus all two-way, three-way, and four-way interactions between condition, mapping, expertise, and half.

**Key hypothesis tests:**

1. `condition` main effect: Is there a significant difference between Tracking and Playback across all conditions? This tests whether the neural population as a whole encodes the behavioral distinction between active frequency control and passive listening.

2. `condition:mapping` interaction: Does the Track-to-Playback neural difference depend on whether a mapping change occurred? This is the central experimental question — if the Track-vs-PB neural signature reflects the mapping between motor output and sensory feedback, then changing that mapping (MP2) should alter or weaken the Track-vs-PB difference relative to sessions where the mapping is stable (MP1).

3. `condition:mapping:expertise` interaction: Does the effect of mapping change on Track-vs-PB encoding evolve with experience?

**Data structure:**
- Each neuron contributes multiple rows: one for Tracking and one for Playback (paired within-neuron design)
- The same neuron_id appears under both condition values, enabling the random slope for condition
- session_id groups neurons from the same recording session
