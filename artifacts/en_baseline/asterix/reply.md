### Design Overview

#### Core Strategy: Center-Column Vertical Play (`x = 5`) with Sticky-Action Foresight
1. **Stay in column `x = 5` (starting column):**
   - `reset()` always places the player at `(x=5, y=5)` and constrains `y` to rows `1..8`. Disabling `left`, `right`, and `fire` (`b3 = -120`, giving probability $< 10^{-17}$ per step) keeps the player locked at `x = 5` for the entire episode.
   - **Zero spawn-kill risk:** At max difficulty (`spawn_speed = 1`, `move_speed = 1`), newly spawned entities appear at `x = 0` or `9` and move to `x = 1` or `8` within the same `act()` step. Staying at `x = 5` provides 3–4 full steps of advance warning for every enemy while remaining completely immune to edge spawn-kills.
   - **100% gold coverage:** Every treasure spawns at `x = 0` (moving right) or `x = 9` (moving left), so every treasure in the game crosses `x = 5`. Because collision check 1 runs before entities move, a treasure at horizontal distance `d` from `x = 5` is catchable from vertical distance `|y - r| <= d + 1`.
   - **Exact single-entity-per-row linear decoding:** `self.entities` has 8 slots (`y = 1..8`), so each row contains at most one entity at any time. For any desired function of an entity's `(column, direction)` in row `r`, solving the two decoupled 10-variable even/odd parity chains across `(entity[r, c], trail[r, c])` yields an **exact** linear pre-activation in a single Layer-1 unit. Moreover, for both our enemy and treasure detectors, every trail weight is $\le 0$, so gold never triggers enemy units and enemies never trigger gold units.

2. **Sticky-action (`p = 0.1`) survival engineering:**
   - **Early dodging (beats sticky delay):** Waiting until distance `d = 1` gives a 10% chance per enemy that sticky `no-op` kills you. Dodging at `d = 3` (and at `d = 4` when a clean escape exists) reduces sticky-delay death probability to $0.1^3 = 0.001$ (left-moving) and $0.1^4 = 0.0001$ (right-moving).
   - **Multi-step overshoot protection (`r±2`, `r±3`, `r±4`):** After moving `UP` or `DOWN`, sticky action repeats the move into `r±2` with probability `0.1`, into `r±3` with probability `0.01`, and into `r±4` with probability `0.001`. Each move unit penalizes single-overshoot hazards in `r±2` (`d <= 2`), double-overshoot hazards in `r±3` (`d in {2, 3}`), and triple-overshoot hazards in `r±4`. Thus the agent never takes a 10% or 1% overshoot death risk for a single treasure, waits 1 step at `d = 3` if both escapes have overshoot hazards, and immediately leaves wall rows (`r = 1, 8`, which only have one escape) when clear.
   - **Idle stillness & interior bias:** Interior rows (`2..7`) carry a small `NO-OP` stillness preference to prevent unnecessary moves, and escape tie-breaking biases away from walls `1` and `8` toward the center.

---

#### Hidden Unit Allocation

- **Layer 1 (`h1`, 27 active units out of 32; scaled into `[0, 1]` via `(h1 + 1) / 2` in Layer 2):**
  - `h1[0..7]` — **`E_lethal(r)`** (`r = 1..8`): Exact binary detector (`+1` vs `-1`) for enemies lethal on this step (`d in {0, 1}`: `(4,R), (5,R), (5,L), (6,L)`), computed simply as `12*(enemy[4]+enemy[5]+enemy[6]-trail[5]) - 6`.
  - `h1[8..15]` — **`E_soon(r)`** (`r = 1..8`): Exact direction-aware graded detector for approaching enemies: `Soon = 1.00` (`d = 2`), `0.50` (`d = 3`), `0.15` (`d = 4`), and `0.00` for `d in {0, 1}`, passed enemies, empty rows, or gold.
  - `h1[16..23]` — **`G(r)`** (`r = 1..8`): Exact direction-aware graded detector for incoming treasure: `Gold = 1.00` (`d = 0`), `0.85` (`d = 1`), `0.70` (`d = 2`), `0.55` (`d = 3`), `0.45` (`d = 4`), `0.35` (`d = 5`), and `0.00` for passed gold, empty rows, or enemies. Prioritizing smaller `d` lets the agent collect imminent gold first and then return for later gold in neighboring rows.
  - `h1[24..26]` — **`P_bit0, P_bit1, P_bit2`**: 3-bit binary (`±1`) encoding of `player_y` (`r in 1..8`), allowing exact per-row gating in Layer 2 using only 3 Layer-1 units.

- **Layer 2 (`h2`, all 32 units; inactive rows saturated to `-1` and offset-cancelled in `b3`):**
  - `h2[0..7]` — **`U_stay(1..8)`**: Gated on `player_y == r`. Evaluates `NO-OP` from `Lethal(r)`, `Soon(r)`, `Gold(r)`, and interior stillness bias; weighted by `W3 = 80`.
  - `h2[8..14]` — **`U_up(2..8)`**: Gated on `player_y == r` (`UP` is disabled at wall `r = 1`). Evaluates `UP` using target row `r-1` (`Lethal`, `Soon`, `Gold`), overshoot `r-2` (`Lethal`, `Soon`), double-overshoot `r-3` (`Soon`), triple-overshoot `r-4` (`Soon`), and center-gravity bias; weighted by `W3 = 80`.
  - `h2[15..21]` — **`U_down(1..7)`**: Exact mirror of `U_up` for `DOWN` (disabled at wall `r = 8`); weighted by `W3 = 80`.
  - `h2[22..26]` — **`F_up(3..7)`** and `h2[27..31]` — **`F_down(2..6)`**: Saturated far-gold pull (`|k - r| >= 2`) contributing `[0, +7.6]` logits (`W3 = 3.8`) strictly additively in the gold's direction. Explicitly gated off by strong negative weights on `Lethal`/`Soon` in `r±1` and `r±2`, ensuring far gold can pull the agent across the entire board when clear, yet can never override any survival or overshoot constraint.

```edits
# ==============================================================================
# LAYER 1 (W1: 600 x 32, b1: 32)
# Input index i = (row * 10 + col) * 6 + ch
# Channels: 0=player, 1=enemy, 2=gold, 4=trail
# ==============================================================================

# --- Units 0..7: E_lethal(r) for r = 1..8 (enemy at d in {0, 1}) ---
b1[0:8] = -6
W1[61:120:6, 0]  = [0, 0, 0, 0, 12, 12, 12, 0, 0, 0]
W1[94, 0] = -12
W1[121:180:6, 1] = [0, 0, 0, 0, 12, 12, 12, 0, 0, 0]
W1[154, 1] = -12
W1[181:240:6, 2] = [0, 0, 0, 0, 12, 12, 12, 0, 0, 0]
W1[214, 2] = -12
W1[241:300:6, 3] = [0, 0, 0, 0, 12, 12, 12, 0, 0, 0]
W1[274, 3] = -12
W1[301:360:6, 4] = [0, 0, 0, 0, 12, 12, 12, 0, 0, 0]
W1[334, 4] = -12
W1[361:420:6, 5] = [0, 0, 0, 0, 12, 12, 12, 0, 0, 0]
W1[394, 5] = -12
W1[421:480:6, 6] = [0, 0, 0, 0, 12, 12, 12, 0, 0, 0]
W1[454, 6] = -12
W1[481:540:6, 7] = [0, 0, 0, 0, 12, 12, 12, 0, 0, 0]
W1[514, 7] = -12

# --- Units 8..15: E_soon(r) for r = 1..8 (enemy at d=2 -> 1.0, d=3 -> 0.5, d=4 -> 0.15) ---
b1[8:16] = -5
W1[61:120:6, 8]   = [0, 4.13, 5, 14.13, 5, 14.13, 5, 14.13, 5, 4.13]
W1[64:120:6, 8]   = [0, 0, -4.13, -5, -14.13, -5, -14.13, -5, -4.13, 0]
W1[121:180:6, 9]  = [0, 4.13, 5, 14.13, 5, 14.13, 5, 14.13, 5, 4.13]
W1[124:180:6, 9]  = [0, 0, -4.13, -5, -14.13, -5, -14.13, -5, -4.13, 0]
W1[181:240:6, 10] = [0, 4.13, 5, 14.13, 5, 14.13, 5, 14.13, 5, 4.13]
W1[184:240:6, 10] = [0, 0, -4.13, -5, -14.13, -5, -14.13, -5, -4.13, 0]
W1[241:300:6, 11] = [0, 4.13, 5, 14.13, 5, 14.13, 5, 14.13, 5, 4.13]
W1[244:300:6, 11] = [0, 0, -4.13, -5, -14.13, -5, -14.13, -5, -4.13, 0]
W1[301:360:6, 12] = [0, 4.13, 5, 14.13, 5, 14.13, 5, 14.13, 5, 4.13]
W1[304:360:6, 12] = [0, 0, -4.13, -5, -14.13, -5, -14.13, -5, -4.13, 0]
W1[361:420:6, 13] = [0, 4.13, 5, 14.13, 5, 14.13, 5, 14.13, 5, 4.13]
W1[364:420:6, 13] = [0, 0, -4.13, -5, -14.13, -5, -14.13, -5, -4.13, 0]
W1[421:480:6, 14] = [0, 4.13, 5, 14.13, 5, 14.13, 5, 14.13, 5, 4.13]
W1[424:480:6, 14] = [0, 0, -4.13, -5, -14.13, -5, -14.13, -5, -4.13, 0]
W1[481:540:6, 15] = [0, 4.13, 5, 14.13, 5, 14.13, 5, 14.13, 5, 4.13]
W1[484:540:6, 15] = [0, 0, -4.13, -5, -14.13, -5, -14.13, -5, -4.13, 0]

# --- Units 16..23: G(r) for r = 1..8 (incoming gold d=0..5 -> 1.0, 0.85, 0.70, 0.55, 0.45, 0.35) ---
b1[16:24] = -5
W1[62:120:6, 16]  = [4.69, 4.9, 9.79, 10.32, 15.66, 19.32, 15.66, 10.32, 9.79, 4.9]
W1[64:120:6, 16]  = [0, -4.69, -4.9, -9.79, -10.32, -15.66, -10.32, -9.79, -4.9, -4.69]
W1[122:180:6, 17] = [4.69, 4.9, 9.79, 10.32, 15.66, 19.32, 15.66, 10.32, 9.79, 4.9]
W1[124:180:6, 17] = [0, -4.69, -4.9, -9.79, -10.32, -15.66, -10.32, -9.79, -4.9, -4.69]
W1[182:240:6, 18] = [4.69, 4.9, 9.79, 10.32, 15.66, 19.32, 15.66, 10.32, 9.79, 4.9]
W1[184:240:6, 18] = [0, -4.69, -4.9, -9.79, -10.32, -15.66, -10.32, -9.79, -4.9, -4.69]
W1[242:300:6, 19] = [4.69, 4.9, 9.79, 10.32, 15.66, 19.32, 15.66, 10.32, 9.79, 4.9]
W1[244:300:6, 19] = [0, -4.69, -4.9, -9.79, -10.32, -15.66, -10.32, -9.79, -4.9, -4.69]
W1[302:360:6, 20] = [4.69, 4.9, 9.79, 10.32, 15.66, 19.32, 15.66, 10.32, 9.79, 4.9]
W1[304:360:6, 20] = [0, -4.69, -4.9, -9.79, -10.32, -15.66, -10.32, -9.79, -4.9, -4.69]
W1[362:420:6, 21] = [4.69, 4.9, 9.79, 10.32, 15.66, 19.32, 15.66, 10.32, 9.79, 4.9]
W1[364:420:6, 21] = [0, -4.69, -4.9, -9.79, -10.32, -15.66, -10.32, -9.79, -4.9, -4.69]
W1[422:480:6, 22] = [4.69, 4.9, 9.79, 10.32, 15.66, 19.32, 15.66, 10.32, 9.79, 4.9]
W1[424:480:6, 22] = [0, -4.69, -4.9, -9.79, -10.32, -15.66, -10.32, -9.79, -4.9, -4.69]
W1[482:540:6, 23] = [4.69, 4.9, 9.79, 10.32, 15.66, 19.32, 15.66, 10.32, 9.79, 4.9]
W1[484:540:6, 23] = [0, -4.69, -4.9, -9.79, -10.32, -15.66, -10.32, -9.79, -4.9, -4.69]

# --- Units 24..26: 3-bit binary encoding of player row r in 1..8 (channel 0) ---
# r = 1: (-,-,-), 2: (+,-,-), 3: (-,+,-), 4: (+,+,-), 5: (-,-,+), 6: (+,-,+), 7: (-,+,+), 8: (+,+,+)
W1[60:120:6, 24]  = -10
W1[60:120:6, 25]  = -10
W1[60:120:6, 26]  = -10
W1[120:180:6, 24] = 10
W1[120:180:6, 25] = -10
W1[120:180:6, 26] = -10
W1[180:240:6, 24] = -10
W1[180:240:6, 25] = 10
W1[180:240:6, 26] = -10
W1[240:300:6, 24] = 10
W1[240:300:6, 25] = 10
W1[240:300:6, 26] = -10
W1[300:360:6, 24] = -10
W1[300:360:6, 25] = -10
W1[300:360:6, 26] = 10
W1[360:420:6, 24] = 10
W1[360:420:6, 25] = -10
W1[360:420:6, 26] = 10
W1[420:480:6, 24] = -10
W1[420:480:6, 25] = 10
W1[420:480:6, 26] = 10
W1[480:540:6, 24] = 10
W1[480:540:6, 25] = 10
W1[480:540:6, 26] = 10

# ==============================================================================
# LAYER 2 (W2: 32 x 32, b2: 32)
# h1 indices: Lethal(r)=r-1, Soon(r)=r+7, Gold(r)=r+15, P_bits=24..26
# ==============================================================================

# --- Group 1: U_stay(r) for r = 1..8 -> cols 0..7 ---
W2[24:27, 0] = [-20, -20, -20]
W2[0, 0] = -3.0
W2[8, 0] = -0.34
W2[16, 0] = 0.11

W2[24:27, 1] = [20, -20, -20]
W2[1, 1] = -3.0
W2[9, 1] = -0.34
W2[17, 1] = 0.11

W2[24:27, 2] = [-20, 20, -20]
W2[2, 2] = -3.0
W2[10, 2] = -0.34
W2[18, 2] = 0.11

W2[24:27, 3] = [20, 20, -20]
W2[3, 3] = -3.0
W2[11, 3] = -0.34
W2[19, 3] = 0.11

W2[24:27, 4] = [-20, -20, 20]
W2[4, 4] = -3.0
W2[12, 4] = -0.34
W2[20, 4] = 0.11

W2[24:27, 5] = [20, -20, 20]
W2[5, 5] = -3.0
W2[13, 5] = -0.34
W2[21, 5] = 0.11

W2[24:27, 6] = [-20, 20, 20]
W2[6, 6] = -3.0
W2[14, 6] = -0.34
W2[22, 6] = 0.11

W2[24:27, 7] = [20, 20, 20]
W2[7, 7] = -3.0
W2[15, 7] = -0.34
W2[23, 7] = 0.11

b2[0:8] = [-63.23, -63.174, -63.174, -63.174, -63.174, -63.174, -63.174, -63.23]

# --- Group 2: U_up(r) for r = 2..8 -> cols 8..14 ---
# r = 2 (col 8): t1=1
W2[24:27, 8] = [20, -20, -20]
W2[0, 8] = -3.0
W2[8, 8] = -0.32
W2[16, 8] = 0.13

# r = 3 (col 9): t1=2, t2=1
W2[24:27, 9] = [-20, 20, -20]
W2[1, 9] = -3.0
W2[9, 9] = -0.32
W2[17, 9] = 0.13
W2[0, 9] = -0.23
W2[8, 9] = -0.23

# r = 4 (col 10): t1=3, t2=2, t3=1
W2[24:27, 10] = [20, 20, -20]
W2[2, 10] = -3.0
W2[10, 10] = -0.32
W2[18, 10] = 0.13
W2[1, 10] = -0.23
W2[9, 10] = -0.23
W2[8, 10] = -0.20

# r = 5 (col 11): t1=4, t2=3, t3=2, t4=1
W2[24:27, 11] = [-20, -20, 20]
W2[3, 11] = -3.0
W2[11, 11] = -0.32
W2[19, 11] = 0.13
W2[2, 11] = -0.23
W2[10, 11] = -0.23
W2[9, 11] = -0.20
W2[8, 11] = -0.04

# r = 6 (col 12): t1=5, t2=4, t3=3, t4=2
W2[24:27, 12] = [20, -20, 20]
W2[4, 12] = -3.0
W2[12, 12] = -0.32
W2[20, 12] = 0.13
W2[3, 12] = -0.23
W2[11, 12] = -0.23
W2[10, 12] = -0.20
W2[9, 12] = -0.04

# r = 7 (col 13): t1=6, t2=5, t3=4, t4=3
W2[24:27, 13] = [-20, 20, 20]
W2[5, 13] = -3.0
W2[13, 13] = -0.32
W2[21, 13] = 0.13
W2[4, 13] = -0.23
W2[12, 13] = -0.23
W2[11, 13] = -0.20
W2[10, 13] = -0.04

# r = 8 (col 14): t1=7, t2=6, t3=5, t4=4
W2[24:27, 14] = [20, 20, 20]
W2[6, 14] = -3.0
W2[14, 14] = -0.32
W2[22, 14] = 0.13
W2[5, 14] = -0.23
W2[13, 14] = -0.23
W2[12, 14] = -0.20
W2[11, 14] = -0.04

b2[8:15] = [-63.222, -63.66, -63.85, -63.886, -63.88, -63.876, -63.834]

# --- Group 3: U_down(r) for r = 1..7 -> cols 15..21 ---
# r = 1 (col 15): t1=2, t2=3, t3=4, t4=5
W2[24:27, 15] = [-20, -20, -20]
W2[1, 15] = -3.0
W2[9, 15] = -0.32
W2[17, 15] = 0.13
W2[2, 15] = -0.23
W2[10, 15] = -0.23
W2[11, 15] = -0.20
W2[12, 15] = -0.04

# r = 2 (col 16): t1=3, t2=4, t3=5, t4=6
W2[24:27, 16] = [20, -20, -20]
W2[2, 16] = -3.0
W2[10, 16] = -0.32
W2[18, 16] = 0.13
W2[3, 16] = -0.23
W2[11, 16] = -0.23
W2[12, 16] = -0.20
W2[13, 16] = -0.04

# r = 3 (col 17): t1=4, t2=5, t3=6, t4=7
W2[24:27, 17] = [-20, 20, -20]
W2[3, 17] = -3.0
W2[11, 17] = -0.32
W2[19, 17] = 0.13
W2[4, 17] = -0.23
W2[12, 17] = -0.23
W2[13, 17] = -0.20
W2[14, 17] = -0.04

# r = 4 (col 18): t1=5, t2=6, t3=7, t4=8
W2[24:27, 18] = [20, 20, -20]
W2[4, 18] = -3.0
W2[12, 18] = -0.32
W2[20, 18] = 0.13
W2[5, 18] = -0.23
W2[13, 18] = -0.23
W2[14, 18] = -0.20
W2[15, 18] = -0.04

# r = 5 (col 19): t1=6, t2=7, t3=8
W2[24:27, 19] = [-20, -20, 20]
W2[5, 19] = -3.0
W2[13, 19] = -0.32
W2[21, 19] = 0.13
W2[6, 19] = -0.23
W2[14, 19] = -0.23
W2[15, 19] = -0.20

# r = 6 (col 20): t1=7, t2=8
W2[24:27, 20] = [20, -20, 20]
W2[6, 20] = -3.0
W2[14, 20] = -0.32
W2[22, 20] = 0.13
W2[7, 20] = -0.23
W2[15, 20] = -0.23

# r = 7 (col 21): t1=8
W2[24:27, 21] = [-20, 20, 20]
W2[7, 21] = -3.0
W2[15, 21] = -0.32
W2[23, 21] = 0.13

b2[15:22] = [-63.834, -63.876, -63.88, -63.886, -63.85, -63.66, -63.222]

# --- Group 4: F_up(r) for r = 3..7 -> cols 22..26 (safety-gated far-gold pull UP) ---
# r = 3 (col 22): safe(2,1), gold(1)
W2[24:27, 22] = [-30, 30, -30]
W2[1, 22] = -22.5
W2[9, 22] = -22.5
W2[0, 22] = -17.5
W2[8, 22] = -17.5
W2[16, 22] = 5.0

# r = 4 (col 23): safe(3,2), gold(2,1)
W2[24:27, 23] = [30, 30, -30]
W2[2, 23] = -22.5
W2[10, 23] = -22.5
W2[1, 23] = -17.5
W2[9, 23] = -17.5
W2[17, 23] = 5.0
W2[16, 23] = 4.25

# r = 5 (col 24): safe(4,3), gold(3,2,1)
W2[24:27, 24] = [-30, -30, 30]
W2[3, 24] = -22.5
W2[11, 24] = -22.5
W2[2, 24] = -17.5
W2[10, 24] = -17.5
W2[18, 24] = 5.0
W2[17, 24] = 4.25
W2[16, 24] = 4.0

# r = 6 (col 25): safe(5,4), gold(4,3,2,1)
W2[24:27, 25] = [30, -30, 30]
W2[4, 25] = -22.5
W2[12, 25] = -22.5
W2[3, 25] = -17.5
W2[11, 25] = -17.5
W2[19, 25] = 5.0
W2[18, 25] = 4.25
W2[17, 25] = 4.0
W2[16, 25] = 3.75

# r = 7 (col 26): safe(6,5), gold(5,4,3,2,1)
W2[24:27, 26] = [-30, 30, 30]
W2[5, 26] = -22.5
W2[13, 26] = -22.5
W2[4, 26] = -17.5
W2[12, 26] = -17.5
W2[20, 26] = 5.0
W2[19, 26] = 4.25
W2[18, 26] = 4.0
W2[17, 26] = 3.75
W2[16, 26] = 2.5

b2[22:27] = [-167.8, -163.55, -159.55, -155.8, -153.3]

# --- Group 5: F_down(r) for r = 2..6 -> cols 27..31 (safety-gated far-gold pull DOWN) ---
# r = 2 (col 27): safe(3,4), gold(4,5,6,7,8)
W2[24:27, 27] = [30, -30, -30]
W2[2, 27] = -22.5
W2[10, 27] = -22.5
W2[3, 27] = -17.5
W2[11, 27] = -17.5
W2[19, 27] = 5.0
W2[20, 27] = 4.25
W2[21, 27] = 4.0
W2[22, 27] = 3.75
W2[23, 27] = 2.5

# r = 3 (col 28): safe(4,5), gold(5,6,7,8)
W2[24:27, 28] = [-30, 30, -30]
W2[3, 28] = -22.5
W2[11, 28] = -22.5
W2[4, 28] = -17.5
W2[12, 28] = -17.5
W2[20, 28] = 5.0
W2[21, 28] = 4.25
W2[22, 28] = 4.0
W2[23, 28] = 3.75

# r = 4 (col 29): safe(5,6), gold(6,7,8)
W2[24:27, 29] = [30, 30, -30]
W2[4, 29] = -22.5
W2[12, 29] = -22.5
W2[5, 29] = -17.5
W2[13, 29] = -17.5
W2[21, 29] = 5.0
W2[22, 29] = 4.25
W2[23, 29] = 4.0

# r = 5 (col 30): safe(6,7), gold(7,8)
W2[24:27, 30] = [-30, -30, 30]
W2[5, 30] = -22.5
W2[13, 30] = -22.5
W2[6, 30] = -17.5
W2[14, 30] = -17.5
W2[22, 30] = 5.0
W2[23, 30] = 4.25

# r = 6 (col 31): safe(7,8), gold(8)
W2[24:27, 31] = [30, -30, 30]
W2[6, 31] = -22.5
W2[14, 31] = -22.5
W2[7, 31] = -17.5
W2[15, 31] = -17.5
W2[23, 31] = 5.0

b2[27:32] = [-153.3, -155.8, -159.55, -163.55, -167.8]

# ==============================================================================
# LAYER 3 (W3: 32 x 6, b3: 6)
# Actions: 0=no-op, 1=left, 2=up, 3=right, 4=down, 5=fire
# ==============================================================================
W3[0:8, 0]   = 80
W3[8:15, 2]  = 80
W3[22:27, 2] = 3.8
W3[15:22, 4] = 80
W3[27:32, 4] = 3.8
b3[0:6] = [560, -120, 499, -120, 499, -120]
```