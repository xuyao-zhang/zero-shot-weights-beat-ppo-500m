# Third-Party Notices & Attributions

This repository releases original Gemini 4 Neon (`xhigh`) zero-shot weight networks, verification scripts, analysis, and figures while strictly respecting upstream licenses and attributions.

## External Runtime Dependency: `MinAtar==1.0.15` (`GPL-3.0`)
- **Package:** [`MinAtar==1.0.15`](https://github.com/kenjyoung/MinAtar) (Kenny Young & Tian Tian, `arXiv:1903.03176`)
- **License:** GNU General Public License v3.0 (`GPL-3.0`)
- **Status:** Declared strictly as an **unbundled external PyPI runtime dependency** in [`requirements.txt`](requirements.txt). This repository does **not** bundle or redistribute `MinAtar` source code or verbatim upstream prompts containing embedded `MinAtar` Python files.

## Upstream Benchmarks & Papers Cited
1. **MIT CSAIL — *Sparks of Weight-Space Programming*** (Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab, Nov 2026): [Blog](https://rikiyat.github.io/blog/weight-space-programming/) | [Logs Repo (`CC BY 4.0` / `MIT` / `GPL-3.0`)](https://github.com/RikiyaT/lm_understand-logs) — Experiment 5 protocol, `PPO-500M` / `PPO-100M` / `Random` / `GPT-6 Astra` reference baselines.
2. **Shawn Hu (`shawnghu`) — `Claude Fable 5.2` Weight-Space Programming Replication** (Nov 10, 2026): [Hugging Face Dataset (`CC BY 4.0` / `MIT` / `GPL-3.0`)](https://huggingface.co/datasets/shawnghu/fable-weight-space-programming) — `Claude Fable 5.2 (high)` reference (`62.34` WWI / `129.44` WSB Median).
3. **Xingyu Dang (`@xingyudang`) — Writing Weights Index (WWI)** (Nov 2, 2026): [X Post](https://x.com/xingyudang/status/2117428789068386630) — WWI formula (`mean of 100 * q / (1 + q)`) and `Gemini 4 Carbon` (`57.7`) reference.
4. **`eomcaleb/weight-space-bench`** (`MIT` code / `CC BY 4.0` & `GPL-3.0` upstream assets): [GitHub](https://github.com/eomcaleb/weight-space-bench) — WSB PPO-500M median normalized formula and `GPT-6.1 Sol` reference.
5. **Alan Turing Institute** (Goworek et al., `arXiv:2611.21449`, Nov 2026): [arXiv](https://arxiv.org/abs/2611.21449) — `GPT-6 Astra` multi-run reference.
6. **Companion Consolidated Leaderboard**: [`opopile/comprehensive-weight-space-programming-leaderboard`](https://github.com/opopile/comprehensive-weight-space-programming-leaderboard) (`MIT` scripts + `CC BY 4.0` docs/data).
