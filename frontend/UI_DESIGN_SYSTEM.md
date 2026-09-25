# SatQuery Orbit — Design System Specification

## 1. Design Philosophy
**SATQUERY ORBIT** is designed around a single guiding principle:
> **MINIMAL INTERFACE, MAXIMUM VISUAL INTELLIGENCE**

The product blends the conversational elegance of **ChatGPT and Gemini**, the expansive exploration of **Google Earth**, and the forensic precision of a **professional satellite image workstation**.

### What SatQuery Orbit Is:
- Quiet, spacious, and modern
- Evidence-first and trustworthy
- Scientific and fast
- Effortless for general users, powerful on demand for geospatial specialists

### What SatQuery Orbit Is NOT:
- Not a dense military command dashboard
- Not a dark cyberpunk glowing console
- Not a traditional desktop GIS software clone
- Not an admin portal cluttered with benchmark charts

---

## 2. Color Palette & Theming

### Dark Theme (Default)
| Token | Hex / Value | Purpose |
|---|---|---|
| `--bg-primary` | `#0B0D10` | Main application backdrop |
| `--bg-secondary` | `#101318` | Drawers, search bars, group containers |
| `--bg-elevated` | `#15191F` | Result cards, bubbles, modal surfaces |
| `--bg-hover` | `#1A1F27` | Interactive hover states |
| `--bg-active` | `#202631` | Active item selection |
| `--text-primary` | `#F4F6F8` | Primary headlines, conversational text |
| `--text-secondary` | `#A4ACB8` | Subtext, labels, metadata |
| `--text-muted` | `#6F7885` | Timestamps, placeholders, inactive icons |
| `--border-subtle` | `rgba(255, 255, 255, 0.07)` | Standard card and list dividers |
| `--border-medium` | `rgba(255, 255, 255, 0.12)` | Focused controls, active buttons |
| `--border-focus` | `rgba(77, 159, 255, 0.45)` | Input and textarea focus rings |
| `--accent-primary` | `#4D9FFF` | Primary buttons, send action, links |
| `--accent-cyan` | `#38BDF8` | Satellite highlights, water indicators |
| `--status-success` | `#22C55E` | Supported verdict, verified confidence |
| `--status-warning` | `#F59E0B` | Supported with limitations, low confidence |
| `--status-error` | `#EF4444` | Disputed evidence, request failure |

### Light Theme
| Token | Hex / Value | Purpose |
|---|---|---|
| `--bg-primary` | `#F7F8FA` | Main light backdrop |
| `--bg-secondary` | `#F1F3F5` | Light containers and input bars |
| `--bg-elevated` | `#FFFFFF` | Pure white cards and user bubbles |
| `--bg-hover` | `#F5F7FA` | Subtle hover state |
| `--text-primary` | `#17191D` | Deep neutral text |
| `--text-secondary` | `#626A75` | Secondary text |
| `--text-muted` | `#8A919C` | Inactive text and timestamps |
| `--border-subtle` | `#E4E7EB` | Light borders |
| `--accent-primary` | `#2563EB` | Blue accent |
| `--accent-cyan` | `#0284C7` | Satellite teal/cyan |

*Note: Satellite raster imagery, false-color overlays, and spectral masks remain visually identical across Dark and Light themes to preserve radiometric accuracy.*

---

## 3. Surface & Glassmorphism Policy
To avoid visual fatigue and maintain crisp legibility:
- **Glassmorphism (`backdrop-filter: blur(14px)`) is STRICTLY reserved for**:
  1. Floating top navigation header
  2. Floating bottom composer
  3. Image lightbox fullscreen overlays
  4. Slide-out drawer backdrops
- All normal content cards, user message bubbles, and inspector panels use clean, solid elevated neutral surfaces.

---

## 4. Typography Hierarchy
- **Primary UI & Conversational Text**: `Inter`, `-apple-system`, `BlinkMacSystemFont`, `Segoe UI`, `Roboto`
  - Comfortable line heights (`1.5` to `1.65`)
  - Relaxed sentence case; no excessive uppercase headers
  - Regular (`400`), Medium (`500`), Semi-bold (`600`), Bold (`700`)
- **Technical Codes & Metadata**: `JetBrains Mono`, `ui-monospace`, `Menlo`, `monospace`
  - Strictly limited to:
    - Coordinates (`18.5204° N, 73.8567° E`)
    - CRS values (`EPSG:32618`)
    - Result and chat UUIDs (`9538ee99-2953...`)
    - Latency measurements (`4568 ms`)
  - Never used for conversational chat messages or metric descriptions.

---

## 5. Spacing Scale & Border Radii

### Spacing Scale
- `4px` (`--space-1`): Inner chip padding, tight icon margins
- `8px` (`--space-2`): Control gaps, button padding
- `12px` (`--space-3`): Input internal margins, list gaps
- `16px` (`--space-4`): Card padding, drawer edge padding
- `20px` (`--space-5`): Section vertical spacing
- `24px` (`--space-6`): Modal padding, conversational separation
- `32px` (`--space-8`): Message-to-message spacing
- `48px` (`--space-12`): Hero-to-content separation
- `64px` (`--space-16`): Bottom workspace clearance for floating composer

### Border Radii
- `4px`: Small badges, CRS tags
- `8px` (`--radius-sm`): Buttons, search inputs, tool icons
- `12px` (`--radius-md`): Popover menus, recent cards, attachment chips
- `16px` (`--radius-lg`): Result cards, modal boxes, summary panels
- `18px`: User message bubble
- `22px` (`--radius-composer`): Floating conversational composer
- `9999px` (`--radius-full`): Suggestion chips, status pills

---

## 6. Motion & Microinteractions
- Transition duration: `150ms` (fast controls) to `220ms` (drawers, modals).
- Easing: `cubic-bezier(0.16, 1, 0.3, 1)` (natural decelerate).
- Subtle translateY on button hover: `-1px`.
- Modal fade + scale: `from 0.98 to 1.0`.
- All animations respect `prefers-reduced-motion: reduce`.
