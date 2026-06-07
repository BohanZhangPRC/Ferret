---
marp: true
theme: default
size: 16:9
paginate: true
style: |
  section {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #ffffff;
    color: #1a1a2e;
    padding: 45px 55px 35px 55px;
    font-size: 22px;
  }
  section.lead {
    background: linear-gradient(160deg, #f0f9ff 0%, #e0f2fe 40%, #f8fafc 100%);
    text-align: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  h1 { font-size: 1.55em; font-weight: 700; color: #0c4a6e; margin: 0 0 12px 0; }
  h2 { font-size: 1.1em; font-weight: 600; color: #0f5c82; margin: 0 0 10px 0; }
  h3 { font-size: 0.95em; font-weight: 600; color: #0c4a6e; margin: 0 0 6px 0; }
  p { margin: 0 0 6px 0; line-height: 1.45; }
  ul, ol { margin: 4px 0; padding-left: 22px; line-height: 1.45; }
  li { margin-bottom: 3px; }

  .hl { color: #0369a1; font-weight: 600; }
  .hlb { color: #0c4a6e; font-weight: 700; }

  .tag {
    display: inline-block;
    background: #e0f2fe;
    color: #0369a1;
    padding: 1px 8px;
    border-radius: 10px;
    font-size: 0.75em;
    font-weight: 600;
  }

  .cols { display: flex; gap: 28px; }
  .col { flex: 1; }
  .col-2 { flex: 2; }
  .col-narrow { flex: 0.85; }

  .box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
  }
  .box-hl {
    background: #f0f9ff;
    border: 1px solid #bae6fd;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
  }

  .fig {
    border: 2px dashed #cbd5e1;
    border-radius: 8px;
    padding: 14px;
    text-align: center;
    color: #94a3b8;
    font-style: italic;
    font-size: 0.7em;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .num {
    display: inline-flex;
    align-items: center; justify-content: center;
    width: 22px; height: 22px;
    background: #0369a1; color: white;
    border-radius: 50%;
    font-weight: 700; font-size: 0.8em;
    margin-right: 4px; flex-shrink: 0;
  }

  .sm { font-size: 0.72em; color: #64748b; }
  .xs { font-size: 0.6em; color: #94a3b8; }

  .divider { width: 48px; height: 2.5px; background: #7dd3fc; margin: 8px 0 14px 0; }

  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; }

  table { width: 100%; border-collapse: collapse; font-size: 0.68em; }
  th { background: #0c4a6e; color: white; padding: 6px 10px; font-weight: 700; }
  td { padding: 5px 10px; border-bottom: 1px solid #e2e8f0; font-size: 0.95em; }

  .center { text-align: center; }
  .mt { margin-top: 14px; }
  .mb { margin-bottom: 8px; }
---

<!-- _class: lead -->
<!-- _paginate: false -->

# Neural Mechanisms of<br>Sensorimotor Learning<br>in the Auditory Cortex

<div class="mt"></div>

### Bohan Zhang

<span class="tag">LSP, ENS-PSL</span> &nbsp; <span class="sm">Supervised by Dr. Yves Boubenec</span>

<div style="margin-top:32px;" class="xs">
FIRE Doctoral School Interview · 20 min · English
</div>

---

<!-- _class: lead -->

# My Training Path<br>to This Project

<div style="text-align:left;max-width:800px;margin:0 auto;">

<div class="box-hl">

| | |
|---|---|
| **BSc Neuroscience** <span class="sm">Bristol 2018–21</span> | Physiology · Pharmacology · Cellular Neuro |
| **MSc Brain & Mind Sci.** <span class="sm">UCL 2021–24 · Distinction</span> | Neural Imaging & Engineering · <span class="hl">OHBM 2025 Poster</span> |
| **M2 BiP + DEC** <span class="sm">Sorbonne · ENS-PSL 2025–26</span> | Bayesian Brain · Cognitive Models · Comp. Neuro |
| **M2 Internship** <span class="sm">Boubenec Lab, ENS · Current</span> | <span class="hlb">In vivo e-phys</span> · Closed-loop VR · 32ch arrays |

</div>

<div class="box" style="font-size:0.82em;line-height:1.55;">

<span class="hlb">Key trajectory:</span> Cellular neuroanatomy → ML/EEG-fMRI → Bayesian cognition → <span class="hlb">In vivo systems neuroscience</span>

<span class="hlb">Current skills:</span> Chronic multi-electrode recording · Custom acoustic VR · PCA · GPFA · Population decoding · Granger causality

<span class="hlb">Already operating</span> at the exact intersection this PhD demands — <span class="hl">zero preparation time needed</span>

</div>

</div>

---

# The Problem
## Auditory-Motor Coupling

<div class="divider"></div>

<div class="cols">
<div class="col">

**Two fundamental computations** <span class="sm">(Kawato, 1999)</span>

<span class="num">1</span> **Forward Problem** — Predict sensory consequences of actions<br>
<span class="num">2</span> **Inverse Problem** — Convert sensory goals into motor commands

<div class="mt"></div>

**Theoretical backbone: Corollary Discharge**

Motor command copy → sensory cortex **before** movement.<br>
Schneider et al. (2018) *Nature*: mouse A1 develops a **learned filter** for self-generated sounds — abolished when contingency disrupted.

<div class="fig mt" style="height:60px;">
[Figure: Forward/Inverse Model schematic — auditory domain]
</div>

</div>
<div class="col">

**Three Unresolved Questions**

<div class="box-hl">
<span class="hlb">Q1 · Cortical Hierarchy</span><br>
<span class="sm">Are predictions generated in A1 or inherited from higher auditory areas?</span>
</div>

<div class="box-hl">
<span class="hlb">Q2 · Multiple Mappings</span><br>
<span class="sm">How does the brain store & switch between distinct sensorimotor mappings?</span>
</div>

<div class="box-hl">
<span class="hlb">Q3 · Inverse Computation</span><br>
<span class="sm">How are auditory targets converted into motor commands during voluntary production?</span>
</div>

<div class="box" style="font-size:0.78em;">
<b>Model:</b> Ferret — gyrencephalic carnivore with well-characterized auditory hierarchy (A1, AAF, PSF, ADF) and accessible premotor cortex
</div>

</div>
</div>

---

# Preliminary Data
## The Closed-Loop Paradigm

<div class="cols">
<div class="col">

**Real-time:** snout position → acoustic frequency<br>
in head-fixed, freely moving ferrets

**Interleaved design:** <span class="hl">Closed-Loop (Tracking)</span> vs. <span class="hl">Open-Loop Playback</span>

<span class="sm">Identical acoustics — different contingency</span>

<div class="mt"></div>

**Recording:** 32-channel tungsten floating arrays chronically implanted in A1

<span class="sm">Same units tracked across months — full naïve → expert trajectory</span>

<div class="tag mt">First platform for single-neuron auditory-motor learning correlates</div>

</div>
<div class="col-2">

<div class="fig" style="height:240px;">
[Figure 1 — 6 panels from PhD proposal]<br><br>
(A) Closed-loop paradigm &nbsp;(B) Representative trial &nbsp;(C) PSTH aligned to movement onset<br>
(D) A1 firing rate by head speed quintiles &nbsp;(E) Decoding accuracy &nbsp;(F) Noise correlations A1-PMC
</div>

</div>
</div>

---

# Preliminary Data
## Four Key Findings

<div class="grid-2">
<div>

<div class="box-hl">
<span class="hlb">1. Anticipatory Signal</span>

Preparatory activity **200–300 ms before movement onset**, exclusive to closed-loop

Builds progressively: Expert > Intermediate > Beginner

→ Correlate of sensorimotor **learning**, not a fixed A1 property
</div>

<div class="box-hl">
<span class="hlb">3. Sharpened Frequency Tuning</span>

Sensorimotor experience **sculpts sensory representations** in A1

→ Experience-dependent plasticity in primary sensory cortex
</div>

</div>
<div>

<div class="box-hl">
<span class="hlb">2. Kinematic Encoding</span>

A1 firing rate systematically modulated by **head speed**

Population decoding significantly better in closed-loop vs. open-loop <span class="sm">(p < 0.001)</span>

→ A1 encodes **movement parameters** as a consequence of learned contingency
</div>

<div class="box-hl">
<span class="hlb">4. Strengthened A1-PMC Coupling</span>

Noise correlations between **A1 & Premotor Cortex**:<br>
Closed-loop > Open-loop <span class="sm">(p < 0.001)</span>

Increases progressively with learning

→ Consolidation of **premotor-to-auditory functional connectivity**
</div>

</div>
</div>

---

# Working Hypothesis

<div class="box-hl center" style="font-size:0.9em;max-width:880px;margin:0 auto 18px auto;">
Sensorimotor learning <span class="hlb">strengthens correlated activity</span> between PMC neurons encoding <span class="hl">movement</span> and A1 neurons encoding <span class="hl">acoustic frequency</span>, establishing a <span class="hlb">learned forward model</span> in the A1-PMC circuit.
</div>

<div class="cols">
<div class="col">

**The Circuit-Level Mechanism**

<div style="font-size:0.78em;line-height:1.7;">

<span class="num">1</span> PMC → A1 projections carry **corollary discharge**

<span class="num">2</span> Closed-loop experience **potentiates** PMC-A1 synapses encoding the position → frequency mapping

<span class="num">3</span> Motor command → PMC → A1 pre-activates frequency-tuned neurons **BEFORE sound onset**

<span class="num">4</span> **Prediction error** (actual vs. predicted feedback) drives ongoing plasticity

<span class="num">5</span> Multiple mappings → **distinct synaptic patterns**, retrievable by context

</div>

</div>
<div class="col">

<div class="fig" style="height:180px;margin-bottom:12px;">
[Figure: A1-PMC Circuit Schematic]<br>
Motor Command → PMC → Corollary Discharge → A1 Prediction → Compare with Feedback → Plasticity
</div>

<div class="box" style="font-size:0.68em;line-height:1.5;">
<span class="hlb">Testable Predictions</span><br>
(1) Anticipatory signal specific to learned mappings &nbsp;(2) PMC inactivation abolishes the signal<br>
(3) Remapping → old predictions fade, new ones emerge &nbsp;(4) Latent representations of multiple mappings coexist in A1-PMC population space
</div>

</div>
</div>

---

# Three Scientific Goals

<div class="divider"></div>

<div class="grid-3">

<div>

<div class="box-hl">
<span class="hlb" style="font-size:1.05em;">Goal 1</span><br>
<b>Cortical Hierarchy</b>

<div class="mt" style="font-size:0.78em;line-height:1.5;">
Map sensorimotor signal distribution across the auditory cortical hierarchy.

**Approach:**
• Neuropixels 2.0 + 32ch arrays in A1, belt, PMC
• Granger causality & PDC
• Test top-down vs. bottom-up propagation
</div>
</div>

</div>
<div>

<div class="box-hl">
<span class="hlb" style="font-size:1.05em;">Goal 2</span><br>
<b>Multi-Mapping Storage</b>

<div class="mt" style="font-size:0.78em;line-height:1.5;">
Characterize storage & retrieval of multiple position → frequency mappings.

**Approach:**
• 2+ distinct mappings (reversed/orthogonal)
• Probe switching dynamics & latent encoding
• dPCA & latent factor models
</div>
</div>

</div>
<div>

<div class="box-hl">
<span class="hlb" style="font-size:1.05em;">Goal 3</span><br>
<b>Inverse Model</b>

<div class="mt" style="font-size:0.78em;line-height:1.5;">
Reveal how auditory targets are converted into motor commands.

**Approach:**
• Target-frequency reward task
• Encoding models: sensory vs. motor coordinates
• GPFA: goal-directed vs. passive dynamics
</div>
</div>

</div>
</div>

<div class="center mt">
<span class="tag">Neuropixels 2.0</span> <span class="tag">Closed-loop VR</span> <span class="tag">dPCA · GPFA</span> <span class="tag">Granger Causality</span> <span class="tag">Population Decoding</span>
</div>

---

# Project Timeline
## 48-Month PhD Plan

<table>
<thead>
<tr>
  <th style="width:16%;">Phase</th>
  <th style="width:21%;">Year 1 (M1–12)</th>
  <th style="width:21%;">Year 2 (M13–24)</th>
  <th style="width:21%;">Year 3 (M25–36)</th>
  <th style="width:21%;">Year 4 (M37–48)</th>
</tr>
</thead>
<tbody>
<tr>
  <td><b>Goal 1<br>Hierarchy</b></td>
  <td>Surgery standardization<br>Paradigm refinement<br><span class="hl">Multi-area recordings</span></td>
  <td>Directed connectivity<br>Granger causality<br><span class="tag" style="font-size:0.85em;">Manuscript 1</span></td>
  <td></td>
  <td></td>
</tr>
<tr>
  <td><b>Goal 2<br>Multi-Mapping</b></td>
  <td>Multi-mapping training begins</td>
  <td>Behavioral training<br>Storage/retrieval recordings</td>
  <td>dPCA & latent factors<br><span class="tag" style="font-size:0.85em;">Manuscript 2</span></td>
  <td></td>
</tr>
<tr>
  <td><b>Goal 3<br>Inverse Model</b></td>
  <td></td>
  <td>Task design</td>
  <td>Task training<br>Inverse model recordings</td>
  <td>Encoding models<br><span class="tag" style="font-size:0.85em;">Manuscript 3</span></td>
</tr>
<tr>
  <td><b>Integration</b></td>
  <td></td>
  <td><span class="sm">Conferences M10–45<br>SfN · Cosyne · ARO</span></td>
  <td>Conferences continued</td>
  <td>Thesis writing<br><span class="hlb">PhD Defense</span></td>
</tr>
</tbody>
</table>

<div class="xs center mt">Progressive autonomy: close guidance Year 1 → independent leadership Year 3–4 · Weekly one-to-one supervision throughout</div>

---

# Contribution to SDGs
## & Planetary Health

<div class="cols">
<div class="col">

<div class="box-hl">
<span class="hlb">SDG 3 — Good Health & Well-Being</span>

<div style="font-size:0.78em;line-height:1.55;">

Speech & language disorders share a **common mechanistic root**: breakdown in the auditory-motor circuits that predict, monitor, and correct self-generated sounds.

BCI design depends on understanding how **sensory cortices encode** self-generated movement.

→ Implications for disorders affecting **hundreds of millions** globally.

</div>
</div>

<div class="box" style="font-size:0.78em;line-height:1.55;">
<span class="hlb">SDG 4 · 9 · 17</span><br>
**Open methods & datasets** → global scientific capacity<br>
**Neuropixels + dPCA/GPFA** → transferable to international BCI R&D and global brain initiatives<br>
**Cross-border collaboration** → bilateral talent exchange between European and international neuroscience communities
</div>

<div class="box-hl" style="font-size:0.72em;line-height:1.5;">
<span class="hlb">Post-2030:</span> The SDG framework captures health outcomes but not the **upstream investment in basic neuroscience** that makes those outcomes possible. We need an explicit goal protecting **long-term fundamental research as a global commons**.
</div>

</div>
<div class="col">

<div class="box-hl">
<span class="hlb">Planetary Health Perspective</span>

<div style="font-size:0.78em;line-height:1.55;">

**Interdisciplinary research as a template:** bridging single-neuron mechanisms with complex behavioral phenomena — the same integrative thinking needed for planetary challenges.

**Global neurological disorder burden** demands cross-border scientific collaboration beyond isolated disciplinary silos.

**Ethical AI for health:** insights from biological forward/inverse models inspire interpretable, sample-efficient neural network architectures.

</div>
</div>

<div class="box">
<span class="hlb">Responsible Research Framing</span>

<div style="font-size:0.72em;line-height:1.5;">

<span class="hl">Best case:</span> Mechanistic knowledge informing neuroprosthetic & rehabilitative devices for speech/motor disorders.

<span class="hl">Key risk:</span> Over-interpretation of animal findings in clinical contexts — mitigated by precise language about model limitations.

<span class="hl">Governance:</span> Analysis pipelines & workflows publicly released under FAIR principles. All data shared openly.

</div>
</div>

</div>
</div>

---

<!-- _class: lead -->

# Why FIRE &<br>Learning Planet Institute

<div style="text-align:left;max-width:820px;margin:0 auto;">

<div class="box-hl" style="font-size:0.78em;line-height:1.65;">

<span class="hlb" style="font-size:1em;">Three Reasons — FIRE over a Standard Doctoral School</span>

<span class="num">1</span> **Scientific Interdisciplinarity** — This project sits at the intersection of systems neuroscience, computational analysis, and cognitive science of learning — a profile that maps onto FIRE's interdisciplinary mission.

<span class="num">2</span> **Methodological Innovation** — Combines high-density e-phys, closed-loop VR, and computational population analyses — bridging experimental and theoretical domains in ways unusual for a standard doctoral school.

<span class="num">3</span> **Long-Term Responsible Framing** — FIRE requires explicit attention to downstream ethical & societal implications. CIRP workshops + philosophy of science provide space to revisit planetary implications at key milestones.

</div>

<div class="cols">
<div class="col">

<div class="box">
<span class="hlb">Why LPI</span>
<div style="font-size:0.75em;line-height:1.5;">
Planetary-scale thinking applied to learning & cognition — a unique intellectual home for research that <span class="hlb">refuses disciplinary silos</span>.
</div>
</div>

<div class="box-hl">
<span class="hlb">Why I Am Ready Now</span>
<div style="font-size:0.7em;line-height:1.55;">
<span class="hl">Already operating</span> the closed-loop paradigm → zero ramp-up<br>
<span class="hl">Fully trained</span> in lab's analysis pipelines<br>
<span class="hl">Seamless M2 → PhD</span> transition → immediate productivity
</div>
</div>

</div>
<div class="col">

<div class="box-hl">
<span class="hlb">Unique Profile</span>
<div style="font-size:0.7em;line-height:1.55;">
<span class="hl">In vivo e-phys</span> + ML + computation <span class="tag">OHBM 2025</span><br>
<span class="hl">4 animals, 7 hemispheres</span> → publishable data from Month 1<br>
Progressive autonomy: close guidance Year 1 → independent leadership Year 3–4
</div>
</div>

</div>
</div>

</div>

---

<!-- _class: lead -->
<!-- _paginate: false -->

# Thank You

<div class="divider" style="margin:20px auto;"></div>

<div class="box-hl" style="font-size:0.78em;line-height:1.55;max-width:750px;margin:0 auto;text-align:left;">
<span class="hlb">Long-Term Vision</span><br>
Contribute to global brain initiatives & BCI development → build lasting international research bridges → translate fundamental mechanisms into clinical tools for communication disorders worldwide.
</div>

<div style="margin-top:40px;font-size:0.78em;line-height:1.8;">
<b>Bohan Zhang</b><br>
<span class="sm">bohan.zhang@etu.sorbonne-universite.fr</span><br>
<span class="sm">Laboratoire des Systèmes Perceptifs (LSP), ENS-PSL</span>
</div>
