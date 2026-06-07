# FIRE_noCSC_ppt169_20260516 - Design Spec

## I. Project Information

| Item | Value |
| ---- | ----- |
| **Project Name** | FIRE_noCSC_ppt169_20260516 |
| **Canvas Format** | PPT 16:9 (1280×720) |
| **Page Count** | 11 |
| **Design Style** | minimalist academic, Top Consulting mode |
| **Target Audience** | FIRE Doctoral School international scientific jury |
| **Use Case** | 10-minute PhD interview presentation |
| **Created Date** | 2026-05-16 |

## II. Canvas Specification

| Property | Value |
| -------- | ----- |
| **Format** | ppt169 |
| **Dimensions** | 1280×720 |
| **viewBox** | `0 0 1280 720` |
| **Margins** | left/right 60px, top/bottom 50px |
| **Content Area** | 1160×620 |

## III. Visual Theme

- **Style**: minimalist academic, Top Consulting
- **Theme**: Light theme
- **Tone**: professional, clean, modern, scientific

### Color Scheme

| Role | HEX | Purpose |
| ---- | --- | ------- |
| **Background** | `#FFFFFF` | Page background |
| **Secondary bg** | `#F0F9FF` | Card background, highlighted sections |
| **Primary** | `#0C4A6E` | Titles, key sections |
| **Accent** | `#0369A1` | Data highlights, key terms |
| **Secondary accent** | `#7DD3FC` | Borders, dividers |
| **Body text** | `#1A1A2E` | Main body text |
| **Secondary text** | `#64748B` | Captions, annotations |
| **Tertiary text** | `#94A3B8` | Footnotes, page numbers |
| **Border/divider** | `#E2E8F0` | Card borders, divider lines |

## IV. Typography System

- **Typography direction**: modern Latin sans
- **Body baseline**: 18px

| Role | Chinese | English | Fallback tail |
| ---- | ------- | ------- | ------------- |
| **Title** | — | Segoe UI | sans-serif |
| **Body** | — | Segoe UI | sans-serif |
| **Emphasis** | — | Segoe UI | sans-serif |
| **Code** | — | Consolas | monospace |

**Per-role font stacks**:
- Title: `"Segoe UI", system-ui, -apple-system, sans-serif`
- Body: `"Segoe UI", system-ui, -apple-system, sans-serif`
- Emphasis: same as Body
- Code: `Consolas, "Courier New", monospace`

### Font Size Hierarchy

| Purpose | Ratio | Size |
| ------- | ----- | ---- |
| Cover title | 2.5x | 45px Bold |
| Page title | 1.5x | 28px Bold |
| Subtitle | 1.2x | 22px SemiBold |
| Body content | 1x | 18px Regular |
| Card title | 1x | 14px Bold |
| Card body | 0.8x | 14px Regular |
| Annotation | 0.7x | 13px Regular |
| Page number | 0.55x | 10px Regular |

## V. Layout Principles

| Element | Value |
| ------- | ----- |
| Safe margin | 60px |
| Block gap | 32px |
| Card gap | 24px |
| Card padding | 28px |
| Card border radius | 12px |

### Layout Patterns Used

- Slide 1: Single column centered (cover)
- Slide 2: Asymmetric split (4:6) table
- Slide 3: Top-bottom split + 2-column
- Slide 4-5: Asymmetric split + 2×2 matrix
- Slide 6: Top-bottom + symmetric split
- Slide 7: Three column cards
- Slide 8: Full-width table
- Slide 9: Symmetric split (5:5)
- Slide 10: Asymmetric split (5:5)
- Slide 11: Single column centered (closing)

## VI. Icon Usage

Minimal. SDG page uses numbered badges (3, 4, 9, 17). Numbered circles for key findings and goals.

## VII. Visualization

No charts. Timeline as table. Placeholder rectangles for figures on slides 3, 4, 6.

## VIII. Image Usage

No AI images. Placeholder rectangles (dashed border, light fill, italic gray text) on slides 3, 4, 6.

## IX. Content Outline

1. Title — Neural Mechanisms of Sensorimotor Learning in the Auditory Cortex
2. Training Path — BSc→MSc→M2→Internship table
3. The Problem — Forward/Inverse + 3 Questions
4. Preliminary Data — Paradigm description + Figure 1 placeholder
5. Preliminary Data — Four Key Findings (2×2 grid)
6. Working Hypothesis — Circuit mechanism + Figure placeholder
7. Three Scientific Goals — 3-column cards
8. Project Timeline — 48-month table
9. SDGs & Planetary Health — Two columns
10. Why FIRE & LPI — Two columns + cards
11. Thank You — Vision + contact

## X. Speaker Notes

Minimal notes — spoken script exists in separate script.md.

## XI. Tech Constraints

- SVG viewBox: 0 0 1280 720
- No external font loading needed (system fonts)
- All text as SVG `<text>` elements
- Card rectangles with rx=12 for rounded corners
- Page numbers at bottom-right
