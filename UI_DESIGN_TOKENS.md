# SatQuery Liquid Glass — Design Tokens Specification

## 1. Color Palette

### Base Surfaces
| Token | Light Theme Value | Dark Theme Value | Usage |
|---|---|---|---|
| `--bg-base` | `#FAFBFF` | `#080B12` | Full-screen app background |
| `--glass-soft` | `rgba(255, 255, 255, 0.58)` | `rgba(20, 25, 35, 0.55)` | Secondary chips, badges, input icons |
| `--glass-medium` | `rgba(255, 255, 255, 0.62)` | `rgba(20, 25, 35, 0.62)` | Suggestion chips, user bubbles |
| `--glass-strong` | `rgba(255, 255, 255, 0.78)` | `rgba(25, 30, 42, 0.78)` | Active navigation tabs, avatars |
| `--glass-nav` | `rgba(255, 255, 255, 0.55)` | `rgba(15, 18, 28, 0.65)` | Floating navigation pills |
| `--glass-composer` | `rgba(255, 255, 255, 0.64)` | `rgba(18, 22, 32, 0.72)` | Floating bottom input bar |

### Borders
| Token | Light Theme Value | Dark Theme Value | Usage |
|---|---|---|---|
| `--border-glass` | `rgba(255, 255, 255, 0.80)` | `rgba(255, 255, 255, 0.10)` | Primary glass card outlines |
| `--border-secondary` | `rgba(110, 130, 180, 0.12)` | `rgba(130, 150, 200, 0.12)` | Subtle item dividers, input borders |
| `--border-subtle` | `rgba(255, 255, 255, 0.75)` | `rgba(255, 255, 255, 0.08)` | Floating navigation container borders |

### Typography Colors
| Token | Light Theme Value | Dark Theme Value | Usage |
|---|---|---|---|
| `--text-primary` | `#10131A` | `#F5F7FB` | Main headlines, body copy, card labels |
| `--text-secondary` | `#4F5D78` | `#A4ACB8` | Navigation items, secondary descriptions |
| `--text-muted` | `#7E8AA5` | `#6F7885` | Placeholders, timings, timestamps |
| `--text-subtitle` | `#4D5B75` | `#8A94A8` | Hero subtitle, header subtitles |

### Brand Accents & Gradients
| Token | Value | Usage |
|---|---|---|
| `--blue-main` | `#1689FF` | Primary action color, active tab highlight |
| `--blue-electric` | `#2479FF` | Start of gradient accent |
| `--indigo` | `#5B4BFF` | Intermediate gradient stop |
| `--purple` | `#8B2DFF` | Building footprints, avatar gradient end |
| `--pink` | `#FF7BB7` | Change analysis chip, warm highlights |
| `--green-soft` | `#22C59A` | Vegetation, land cover, positive badges |
| `--orange` | `#FF7847` | Compare images chip, report actions |
| `--gradient-accent` | `linear-gradient(90deg, #1889FF, #4778FF, #7A45FF, #9B30FF)` | "Understand Change" and "SatQuery" gradient text |
| `--gradient-send` | `linear-gradient(135deg, #16A0FF, #267DFF, #703BFF, #8C26F5)` | Circular send button |
| `--gradient-avatar` | `linear-gradient(135deg, #2479FF, #8B2DFF)` | User avatar badge |

---

## 2. Elevation & Shadows

| Token | Light Theme Value | Dark Theme Value |
|---|---|---|
| `--shadow-nav` | `0 8px 35px rgba(79,108,170,0.10), inset 0 1px 0 rgba(255,255,255,0.85)` | `0 8px 35px rgba(0,0,0,0.30), inset 0 1px 0 rgba(255,255,255,0.06)` |
| `--shadow-composer` | `0 20px 50px rgba(75,100,170,0.11), inset 0 1px 0 rgba(255,255,255,0.92)` | `0 20px 50px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.05)` |
| `--shadow-card` | `0 14px 40px rgba(75,95,150,0.09)` | `0 14px 40px rgba(0,0,0,0.25)` |
| `--shadow-chip` | `0 8px 24px rgba(70,90,140,0.08)` | `0 8px 24px rgba(0,0,0,0.20)` |
| `--shadow-send` | `0 10px 28px rgba(89,65,255,0.30)` | `0 10px 28px rgba(89,65,255,0.35)` |
| `--shadow-active-nav` | `0 7px 20px rgba(50,110,255,0.12), inset 0 1px 0 rgba(255,255,255,0.95)` | `0 7px 20px rgba(50,110,255,0.20), inset 0 1px 0 rgba(255,255,255,0.08)` |

---

## 3. Radii & Spacing

### Corner Radii
- Small Elements (`--radius-sm`): `8px`
- Medium Elements / Preview Thumbnails (`--radius-md`): `14px`
- Findings Cards (`--radius-card-sm`): `18px - 20px`
- Assistant / User Cards (`--radius-card`): `24px - 28px`
- Navigation, Composer, Action Chips (`--radius-pill`): `999px`

### Spacing Scale
- `--space-1`: `4px`
- `--space-2`: `8px`
- `--space-3`: `12px`
- `--space-4`: `16px`
- `--space-5`: `20px`
- `--space-6`: `24px`
- `--space-8`: `32px`
- `--space-10`: `40px`
- `--space-12`: `48px`
- `--space-16`: `64px`
- `--space-20`: `80px`

---

## 4. Blur & Frosted Glass Filters

- Navigation Pill: `blur(30px) saturate(140%)`
- Floating Composer: `blur(34px) saturate(150%)`
- Result Cards: `blur(26px) saturate(145%)`
- Lightbox Overlays: `blur(8px)`
