### Strategy

1. **Column-locked right-facing posture (`sub_front = col 2`, `sub_back = col 1`, facing right):**
   - At the start of each episode in row 0 (where no enemies spawn and oxygen does not drain), the policy moves horizontally to `(front=2, back=1)` facing right. Once aligned, left/right logits are suppressed to `-600`, so even under 10% sticky actions the submarine stays permanently locked on column 2 facing right.
   - Column 2 is optimal for sticky-action survival: enemies spawning at column 0 need 5–11 steps (even at maximum difficulty `move_speed = 2`) to reach column 2 (`0.1^5` dodge failure rate), enemy bullets coming from the right are detected up to column 6 (4 steps warning, `0.1^4` failure rate), columns 6–9 remain open for rapid shooting (catching both left-moving enemies early before subs fire and right-moving enemies from behind), and every diver crosses column 2 and stays there for 6 steps.
2. **Graded multi-tier danger avoidance:**
   - Each underwater row `r = 1..8` gets a single monotonic danger score `D[r]` distinguishing: safe / enemy at col 6, enemy bullet at col 6 (`ch1=1, ch2=0`), threat at col 0 or 5, threat at col 4, threat at col 1 or 3, and instant-death threat at col 2.
   - Moving up, staying, and moving down are scored primarily by `-D[target_row]`, so the sub evacuates early when safe neighbors exist, waits instead of stepping into a worse threat when boxed in, and escapes into the least-imminent threat when trapped.
3. **Rapid multi-row firing & cooldown pipelining:**
   - Hunting operates in rows `3..8` (keeping row 2 as a safety buffer against sticky-`'u'` accidental surfacing with 0 divers, and gently centering toward row 5 when idle).
   - For each row `3..8`, an enemy detector counts shootable enemies in cols `6..9` (`ch1 + ch2`) **minus** friendly bullets already in flight in cols `3..8` (`ch3`). Immediately after firing at a single enemy, that row's target turns off so the sub moves to the next target row during the 5-step shot cooldown; if a second enemy is present, it stays and fires again.
   - Distance-decayed geometric target weights (`0.145, 0.072, 0.036, 0.020, 0.012`) enforce strict nearest-row pursuit without oscillation.
4. **Diver buffer & safe surfacing:**
   - Because `ch5` overlays `oxygen_guage` (left-aligned) and `diver_guage` (cols `9-d..8`), `ch5[9, 2] == 0` unambiguously means `oxygen < 60`, `ch5[9, 8] == 1` then guarantees `diver_count >= 1`, and `ch5[9, 8]`/`ch5[9, 7]` detect `diver_count < 2` once `oxygen < 180`/`160`.
   - The policy actively intercepts divers in rows `3..8` (cols `1..3`) whenever `diver_count < 2` (maintaining a 1–2 diver reserve while avoiding unnecessary chasing), surfaces cleanly once `oxygen < 60` and `diver_count >= 1` (pausing for unsafe rows above, with an emergency push if `oxygen < 20`), hard-blocks entering row 0 without a confirmed diver, and waits for free at row 0 until rows 1–2 are safe before diving.

---

### Hidden-Unit Layout

- **Layer 1 (`h1`, 32 units):**
  - `h1[0..8]` (`P[0..8]`): Crisp `+1 / -1` indicator of whether the submarine is in row `y` at col 2 (`ch0[y, 2]`).
  - `h1[9..16]` (`D[1..8]`): Graded row danger for rows `1..8` from `ch1[r, 0..6]` and `-ch2[r, 6]` (separating bullets from enemies at col 6).
  - `h1[17]` (`A_raw`): Signed horizontal alignment signal (`+1` move right, `-1` move left, `0` aligned at `front=2, back=1`).
  - `h1[18]` (`M_surf`): Surface mode (`oxygen < 60` and `diver_count >= 1`, from `ch5[9, 8]` and `-ch5[9, 2]`).
  - `h1[19]` (`M_emerg`): Emergency surface mode (`oxygen < 20` and `diver_count >= 1`, from `ch5[9, 8]` and `-ch5[9, 0]`).
  - `h1[20..25]` (`E[3..8]`): Unshot shootable enemy in cols `6..9` of rows `3..8`, subtracting friendly bullets in cols `3..8`.
  - `h1[26..31]` (`V[3..8]`): Catchable diver in cols `1..3` (`ch2 - ch1`) of rows `3..8`, gated by `diver_count < 2` (`ch5[9, 7..8]`).

- **Layer 2 (`h2`, 32 units, row-gated via `6 * P[y] - 6` so inactive rows saturate to `-1`):**
  - `h2[0]` (`A`): Pass-through of horizontal alignment `A_raw` driving `left`/`right` logits.
  - `h2[1..8]` (`U[1..8]`): Value of moving `up` from row `y = 1..8` (danger of `y-1`, surfacing/emergency boost, hard-block at row 1 and buffer penalty at rows 2–3 when not surfacing, plus geometric enemy/diver pull above).
  - `h2[9..16]` (`D_act[0..7]`): Value of moving `down` from row `y = 0..7` (danger of `y+1`, row-0 dive lookahead into rows 1–2, return-to-hunting boost from rows 1–2, surface suppression, and geometric enemy/diver pull below).
  - `h2[17..25]` (`S[0..8]`): Value of staying in row `y = 0..8` (safety of row `y`, plus stay-for-diver bonus on rows `3..8`).
  - `h2[26..31]` (`F[3..8]`): Crisp `+1 / -1` fire gate active iff in row `y` with an unshot enemy (`P[y]` and `E[y]`), adding `+100` to `fire` over `no-op` when active (and `-10` when inactive to avoid wasting shot cooldown).

```edits
# === LAYER 1: P[0..8] (player at col 2 in row y) ===
W1[12, 0] = 10
W1[72, 1] = 10
W1[132, 2] = 10
W1[192, 3] = 10
W1[252, 4] = 10
W1[312, 5] = 10
W1[372, 6] = 10
W1[432, 7] = 10
W1[492, 8] = 10
b1[0:9] = -5

# === LAYER 1: D[1..8] (graded row danger on cols 0..6, distinguishing col-6 bullets vs enemies) ===
W1[61:103:6, 9] = [2.6, 3.7, 5.5, 3.7, 3.1, 2.6, 1.9]
W1[98, 9] = -1
W1[121:163:6, 10] = [2.6, 3.7, 5.5, 3.7, 3.1, 2.6, 1.9]
W1[158, 10] = -1
W1[181:223:6, 11] = [2.6, 3.7, 5.5, 3.7, 3.1, 2.6, 1.9]
W1[218, 11] = -1
W1[241:283:6, 12] = [2.6, 3.7, 5.5, 3.7, 3.1, 2.6, 1.9]
W1[278, 12] = -1
W1[301:343:6, 13] = [2.6, 3.7, 5.5, 3.7, 3.1, 2.6, 1.9]
W1[338, 13] = -1
W1[361:403:6, 14] = [2.6, 3.7, 5.5, 3.7, 3.1, 2.6, 1.9]
W1[398, 14] = -1
W1[421:463:6, 15] = [2.6, 3.7, 5.5, 3.7, 3.1, 2.6, 1.9]
W1[458, 15] = -1
W1[481:523:6, 16] = [2.6, 3.7, 5.5, 3.7, 3.1, 2.6, 1.9]
W1[518, 16] = -1
b1[9:17] = -2.5

# === LAYER 1: A_raw (h1[17], horizontal alignment to front=2, back=1) ===
W1[0:600:60, 17] = 5
W1[16, 17] = 5
W1[18:600:60, 17] = -5
W1[24:600:60, 17] = -5
W1[30:600:60, 17] = -5
W1[36:600:60, 17] = -5
W1[42:600:60, 17] = -5
W1[48:600:60, 17] = -5
W1[54:600:60, 17] = -5
b1[17] = 0

# === LAYER 1: M_surf (h1[18], oxy<60 & diver>=1) and M_emerg (h1[19], oxy<20 & diver>=1) ===
W1[593, 18] = 10
W1[557, 18] = -10
b1[18] = -5
W1[593, 19] = 10
W1[545, 19] = -10
b1[19] = -5

# === LAYER 1: E[3..8] (h1[20..25], unshot enemies in cols 6..9 minus friendly bullets in cols 3..8) ===
W1[217:241:6, 20] = 6
W1[218:242:6, 20] = 6
W1[201:237:6, 20] = -12
W1[277:301:6, 21] = 6
W1[278:302:6, 21] = 6
W1[261:297:6, 21] = -12
W1[337:361:6, 22] = 6
W1[338:362:6, 22] = 6
W1[321:357:6, 22] = -12
W1[397:421:6, 23] = 6
W1[398:422:6, 23] = 6
W1[381:417:6, 23] = -12
W1[457:481:6, 24] = 6
W1[458:482:6, 24] = 6
W1[441:477:6, 24] = -12
W1[517:541:6, 25] = 6
W1[518:542:6, 25] = 6
W1[501:537:6, 25] = -12
b1[20:26] = -9

# === LAYER 1: V[3..8] (h1[26..31], catchable divers in cols 1..3 gated by diver_count < 2) ===
W1[188:206:6, 26] = 12
W1[187:205:6, 26] = -12
W1[248:266:6, 27] = 12
W1[247:265:6, 27] = -12
W1[308:326:6, 28] = 12
W1[307:325:6, 28] = -12
W1[368:386:6, 29] = 12
W1[367:385:6, 29] = -12
W1[428:446:6, 30] = 12
W1[427:445:6, 30] = -12
W1[488:506:6, 31] = 12
W1[487:505:6, 31] = -12
W1[593, 26:32] = -7
W1[587, 26:32] = -6
b1[26:32] = -4

# === LAYER 2: A (h2[0], alignment pass-through) ===
W2[17, 0] = 5
b2[0] = 0

# === LAYER 2: U[1..8] (h2[1..8], move UP from row y) ===
W2[1, 1] = 6
W2[18, 1] = 3.5
b2[1] = -6.5

W2[2, 2] = 6
W2[9, 2] = -1
W2[18, 2] = 0.99
W2[19, 2] = 0.45
b2[2] = -6.86

W2[3, 3] = 6
W2[10, 3] = -1
W2[18, 3] = 0.23
W2[19, 3] = 0.45
b2[3] = -6.1

W2[4, 4] = 6
W2[11, 4] = -1
W2[18, 4] = 0.19
W2[19, 4] = 0.45
W2[20, 4] = 0.0725
W2[26, 4] = 0.16
b2[4] = -5.8275

W2[5, 5] = 6
W2[12, 5] = -1
W2[18, 5] = 0.19
W2[19, 5] = 0.45
W2[20:22, 5] = [0.036, 0.0725]
W2[26:28, 5] = [0.155, 0.16]
b2[5] = -5.6365

W2[6, 6] = 6
W2[13, 6] = -1
W2[18, 6] = 0.19
W2[19, 6] = 0.45
W2[20:23, 6] = [0.018, 0.036, 0.0725]
W2[26:29, 6] = [0.15, 0.155, 0.16]
b2[6] = -5.4645

W2[7, 7] = 6
W2[14, 7] = -1
W2[18, 7] = 0.19
W2[19, 7] = 0.45
W2[20:24, 7] = [0.01, 0.018, 0.036, 0.0725]
W2[26:30, 7] = [0.1475, 0.15, 0.155, 0.16]
b2[7] = -5.307

W2[8, 8] = 6
W2[15, 8] = -1
W2[18, 8] = 0.19
W2[19, 8] = 0.45
W2[20:25, 8] = [0.006, 0.01, 0.018, 0.036, 0.0725]
W2[26:31, 8] = [0.145, 0.1475, 0.15, 0.155, 0.16]
b2[8] = -5.154

# === LAYER 2: D_act[0..7] (h2[9..16], move DOWN from row y) ===
W2[0, 9] = 6
W2[9, 9] = -1
W2[10, 9] = -0.25
b2[9] = -6.7

W2[1, 10] = 6
W2[10, 10] = -1
W2[18, 10] = -0.1
b2[10] = -6.6

W2[2, 11] = 6
W2[11, 11] = -1
W2[18, 11] = -0.225
b2[11] = -6.775

W2[3, 12] = 6
W2[12, 12] = -1
W2[18, 12] = -0.15
W2[21:26, 12] = [0.0725, 0.036, 0.018, 0.01, 0.006]
W2[27:32, 12] = [0.16, 0.155, 0.15, 0.1475, 0.145]
b2[12] = -5.946

W2[4, 13] = 6
W2[13, 13] = -1
W2[18, 13] = -0.15
W2[22:26, 13] = [0.0725, 0.036, 0.018, 0.01]
W2[28:32, 13] = [0.16, 0.155, 0.15, 0.1475]
b2[13] = -6.097

W2[5, 14] = 6
W2[14, 14] = -1
W2[18, 14] = -0.15
W2[23:26, 14] = [0.0725, 0.036, 0.018]
W2[29:32, 14] = [0.16, 0.155, 0.15]
b2[14] = -6.2585

W2[6, 15] = 6
W2[15, 15] = -1
W2[18, 15] = -0.15
W2[24:26, 15] = [0.0725, 0.036]
W2[30:32, 15] = [0.16, 0.155]
b2[15] = -6.4265

W2[7, 16] = 6
W2[16, 16] = -1
W2[18, 16] = -0.15
W2[25, 16] = 0.0725
W2[31, 16] = 0.16
b2[16] = -6.6175

# === LAYER 2: S[0..8] (h2[17..25], STAY in row y) ===
W2[0, 17] = 6
b2[17] = -5.713

W2[1, 18] = 6
W2[9, 18] = -1
b2[18] = -6.7

W2[2, 19] = 6
W2[10, 19] = -1
b2[19] = -6.7

W2[3, 20] = 6
W2[11, 20] = -1
W2[26, 20] = 0.16
b2[20] = -6.54

W2[4, 21] = 6
W2[12, 21] = -1
W2[27, 21] = 0.16
b2[21] = -6.54

W2[5, 22] = 6
W2[13, 22] = -1
W2[28, 22] = 0.16
b2[22] = -6.54

W2[6, 23] = 6
W2[14, 23] = -1
W2[29, 23] = 0.16
b2[23] = -6.54

W2[7, 24] = 6
W2[15, 24] = -1
W2[30, 24] = 0.16
b2[24] = -6.54

W2[8, 25] = 6
W2[16, 25] = -1
W2[31, 25] = 0.16
b2[25] = -6.54

# === LAYER 2: F[3..8] (h2[26..31], fire gate for rows 3..8) ===
W2[3, 26] = 6
W2[20, 26] = 4
W2[4, 27] = 6
W2[21, 27] = 4
W2[5, 28] = 6
W2[22, 28] = 4
W2[6, 29] = 6
W2[23, 29] = 4
W2[7, 30] = 6
W2[24, 30] = 4
W2[8, 31] = 6
W2[25, 31] = 4
b2[26:32] = -6

# === LAYER 3: Output logits (0:n, 1:l, 2:u, 3:r, 4:d, 5:f) ===
W3[17:26, 0] = 400
W3[0, 1] = -1200
W3[1:9, 2] = 400
W3[0, 3] = 1200
W3[9:17, 4] = 400
W3[17:26, 5] = 400
W3[26:32, 5] = 55
b3[0:6] = [3200, -600, 2800, -600, 2800, 3520]
```