## 9. Summary: What This Notebook Shows

This notebook is an Ariel FGS pointability and diversity audit. The question is not simply **how many targets are lost**, but whether the remaining targets still cover the population-level diversity Ariel needs: planet size, planet mass, equilibrium temperature, host metallicity, and host temperature.

### Targets Compared

1. **2026 Known targets**  
   The full `Ariel_MCS_Known_2026-05-11.csv` list, compared against Andrea's two Tier-2 pointable 2000-best scenarios:
   - `2000-best FGS2`: optimistic guiding case
   - `2000-best FGS1`: pessimistic FGS1-only guiding case

2. **Known + TPC targets**  
   The combined Known + TPC catalog (`3440` targets), compared against the same two Andrea/Billy 2000-best pointable lists. This asks whether the 2000-best survey design recovers the diversity available in the larger future target pool.

3. **Brightest 500 Known+TPC targets**  
   The 500 targets with the lowest stellar `Ks` magnitude, used as a high-SNR backbone diagnostic. Here we explicitly compare:
   - full Brightest 500
   - Brightest 500 under FGS2 guiding
   - Brightest 500 under FGS1-only guiding

### Diversity Axes

The notebook makes heatmaps for four projections of the diversity space:

- planet radius vs. equilibrium temperature
- planet mass vs. equilibrium temperature
- host temperature vs. host metallicity
- planet radius vs. host temperature

It also computes:

- one-axis diversity retention
- full 5D endangered-cell audits
- normalized Cowan-style leverage in log planet mass and stellar metallicity
- science-category retention for the cells most relevant to Ariel's population science

---

## Headline Results

### 1. Known Targets: FGS1-only Is Much Harsher Than FGS2

For the Known-target comparison:

| Scenario | N targets |
|---|---:|
| 2026 Known | 977 |
| 2000-best FGS2 | 593 |
| 2000-best FGS1 | 380 |

The 2000-best FGS2 case keeps more targets and more diversity than the 2000-best FGS1 case. The largest FGS1-only losses are in the faint/cool-host population cells:

| Science category | 2026 Known | 2000-best FGS2 | 2000-best FGS1 |
|---|---:|---:|---:|
| M-dwarf hosts | 131 | 42 (32%) | 10 (8%) |
| Small planets around M dwarfs | 108 | 40 (37%) | 9 (8%) |
| Temperate small planets | 216 | 63 (29%) | 52 (24%) |
| Temperate rocky/super-Earths | 152 | 47 (31%) | 30 (20%) |
| Hot giants | 475 | 442 (93%) | 236 (50%) |

**Interpretation:** the 2000-best FGS1-only case does not just reduce sample size. It preferentially removes the cool-star and small-planet diversity that Ariel needs for population-level science beyond hot giants.

### Plot: Known Targets

![Known targets: planet radius vs. equilibrium temperature](diversity_known_rp_teq.png)

---

### 2. Known + TPC: The Larger Future Pool Contains Diversity, But the 2000-Best Lists Do Not Fully Preserve It

The full Known+TPC universe has `3440` targets. It is much richer than the current Known list, especially for future small-planet science. However, the Andrea/Billy 2000-best lists are only subsets of that larger pool:

| Scenario | N targets |
|---|---:|
| Known+TPC | 3440 |
| 2000-best FGS2 | 593 |
| 2000-best FGS1 | 380 |

Relative to the full Known+TPC catalog:

| Science category | Known+TPC | 2000-best FGS2 | 2000-best FGS1 |
|---|---:|---:|---:|
| M-dwarf hosts | 219 | 42 (19%) | 10 (5%) |
| Small planets around M dwarfs | 137 | 40 (29%) | 9 (7%) |
| Temperate small planets | 318 | 63 (20%) | 52 (16%) |
| Temperate rocky/super-Earths | 189 | 47 (25%) | 30 (16%) |
| Hot giants | 2121 | 442 (21%) | 236 (11%) |

**Interpretation:** the full Known+TPC catalog contains a much broader future science pool, but the 2000-best pointable lists do not automatically preserve that diversity. They should be judged by whether they repopulate endangered cells, not only by total target count or Tier-2 efficiency.

### Plot: Known + TPC Targets

![Known + TPC targets: planet radius vs. equilibrium temperature](diversity_kpt_rp_teq.png)

---

### 3. Brightest 500: The High-SNR Backbone Mostly Survives

The Brightest 500 diagnostic is the most reassuring result. These are the easiest high-SNR targets in the Known+TPC catalog.

| Scenario | N targets |
|---|---:|
| Brightest 500 | 500 |
| Brightest 500 FGS2 | 499 |
| Brightest 500 FGS1 | 481 |

Under FGS1-only guiding, the Brightest 500 retain most of the important science categories:

| Science category | Brightest 500 | FGS2 | FGS1 |
|---|---:|---:|---:|
| Temperate small planets | 200 | 200 (100%) | 184 (92%) |
| Temperate rocky/super-Earths | 128 | 128 (100%) | 112 (88%) |
| Small planets around M dwarfs | 69 | 69 (100%) | 51 (74%) |
| M-dwarf hosts | 70 | 70 (100%) | 52 (74%) |
| C/O-ready hosts | 22 | 22 (100%) | 22 (100%) |
| Hot giants | 113 | 112 (99%) | 112 (99%) |

**Interpretation:** FGS1-only does **not** destroy Ariel's bright high-SNR survey backbone. It still removes some M-dwarf and small-planet systems, but the loss is not catastrophic inside the brightest 500.

### Plot: Brightest 500 Targets

![Brightest 500 targets: planet radius vs. equilibrium temperature](diversity_b500_rp_teq.png)

---

## Leverage Results

The notebook computes normalized survey leverage following the Cowan-style definition:

$$
L = \sqrt{\sum_i \frac{(x_i - \bar{x})^2}{\sigma_{x,i}^2}}.
$$

For planet mass, the axis is $\log_{10}(M_p/M_\oplus)$; for stellar diversity, the axis is host `[Fe/H]`.

Key leverage result:

| Scenario | $L_p$ log mass | $L_s$ [Fe/H] |
|---|---:|---:|
| 2026 Known | 2809 | 101.9 |
| 2000-best FGS2 | 601 | 63.0 |
| 2000-best FGS1 | 530 | 59.2 |
| Brightest 500 | 3628 | 66.0 |
| Brightest 500 FGS2 | 3633 | 66.0 |
| Brightest 500 FGS1 | 3565 | 65.4 |

The Brightest 500 retain nearly all leverage under FGS1-only:

- $L_p$: `3565 / 3628`, about `98%`
- $L_s$: `65.4 / 66.0`, about `99%`

**Interpretation:** the bright target backbone keeps almost all of its population-slope measuring power. The major loss is not leverage in the bright sample; it is diversity in the faint/cool-star tail.

### FGS1 vs FGS2: What gives?

![Normalized survey leverage by scenario](leverage_by_scenario.png)

---

## Science Categories Most Affected

The science-category diagnostic confirms the same story: FGS1-only mainly hurts cool-star and small-planet diversity.

### Known Targets

![Known targets: science categories retained](killer_science_retention_known.png)

### Known + TPC Targets

![Known + TPC targets: science categories retained](killer_science_retention_kpt.png)

### Brightest 500 Targets

![Brightest 500 targets: science categories retained](killer_science_retention_b500.png)

---

## Important Caveat

The `C/O-ready hosts = 0` result for the Andrea/Billy 2000-best files is a metadata limitation, not necessarily a true astrophysical absence. Those reduced pointability files do not carry the stellar C/O columns from the full catalog. The Brightest 500 and full Known+TPC catalog retain those metadata columns, so their C/O-ready counts are more meaningful.

---

## Bottom Line

FGS1-only does **not** kill Ariel's bright, high-SNR population survey. The Brightest 500 remain scientifically strong and retain almost all leverage.

But FGS1-only does preferentially remove the faint/cool-host tail. The science most at risk is:

- M-dwarf planets
- small planets around cool stars
- temperate rocky/super-Earth targets
- faint-host diversity more generally

FGS2 is safer. FGS1-only is survivable for the bright backbone, but damaging for the parts of Ariel's science case that depend on cool, faint, small-planet systems. The 2000-best lists should therefore be evaluated not just by total N or Tier-2 efficiency, but by whether they refill the endangered cells in planet radius, temperature, host metallicity, and host temperature.
