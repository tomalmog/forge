# Forge Studio UI Theme Spec

## Purpose
This document defines the visual language for Forge Studio so future UI work stays consistent.
The target style is inspired by Zed IDE: sharp geometry, low chroma, dense information layout, and minimal decorative effects.

## Visual Goals
- Prioritize legibility for long sessions.
- Keep focus on data and training controls, not decoration.
- Use a restrained accent system where color communicates state.
- Preserve clear affordances for interactive elements.

## Core Principles
- Square first: prefer small radii (`2px` to `6px`) over rounded cards.
- Flat layers: avoid heavy shadows and gradients.
- Pane architecture: UI should feel like docked IDE panes separated by subtle borders.
- Dense but readable: compact spacing, consistent typography scale.
- One primary accent: use blue for active/focus states.

## Color Tokens
All colors should be assigned through CSS variables.

### Base
- `--bg`: window background.
- `--bg-elevated`: raised pane background.
- `--bg-soft`: subtle alternate row or section background.
- `--line`: pane and control borders.

### Text
- `--ink`: primary text.
- `--muted`: secondary text.
- `--muted-strong`: labels and headings with less emphasis than primary text.

### Semantic
- `--accent`: active/focus/selected state.
- `--accent-strong`: accent hover/pressed state.
- `--warning`: warning and destructive actions.

## Typography
- UI text: `Space Grotesk`.
- Dense/technical text and numbers: `IBM Plex Mono`.
- Body sizes should stay in `12px` to `15px` range for dense panes.
- Use stronger weight changes instead of large size jumps.

## Layout Rules
- App uses a three-column shell: dataset sidebar, workspace, views sidebar.
- Each column owns its vertical scroll.
- Sidebars are full-height panes with subtle separators.
- Workspace uses section stacking with compact gaps.

## Component Rules

### Panels
- Background from `--bg-elevated`.
- Border `1px solid var(--line)`.
- Radius `4px` to `6px`.
- No drop shadows in default state.

### Inputs and Selects
- Flat dark background.
- Border only for separation.
- Focus uses accent outline/ring.
- No pill shapes.

### Buttons
- Default button is neutral pane style.
- Primary actions use accent fill.
- Warning actions use warning color.
- Radius `4px`.

### Lists and Cards
- Selected state uses accent-tinted background and border.
- Hover state uses subtle brightness shift.
- Preserve monospace for version IDs and run artifacts.

### Charts
- Dark chart background aligned with panels.
- Thin grid lines with low contrast.
- Axis/tick labels in muted colors.
- Series colors should be limited and consistent:
  - train: accent blue
  - validation: warm contrast (amber)

## Motion
- Keep transitions short (`120ms` to `180ms`).
- Animate only opacity/background/border for hover and panel toggles.
- Avoid large movement animations.

## Accessibility
- Maintain visible keyboard focus on all interactive controls.
- Keep text/background contrast high enough for small UI text.
- Do not use color as the only indicator for selected/active state.

## Rollout Plan
1. Theme foundation: tokens and shell layout styles.
2. Shared controls: panel, input, buttons, list rows.
3. Data views: dashboard cards, diff view, sample inspector.
4. Training views: pipeline canvas, progress bars, charts, console.
5. Polish pass: spacing and alignment consistency.

## Out of Scope
- Functional workflow changes.
- Information architecture changes.
- New navigation patterns.

