# SatQuery Liquid Glass — Frontend Component Architecture Map

## Overview
The **SatQuery Liquid Glass** frontend design combines Apple-like translucence, ChatGPT/Gemini conversational simplicity, and high-precision satellite analysis. It presents an intuitive, uncluttered stage with floating glass pills, interactive before/after sliders, and structured insight matrices over an animated liquid aurora background.

---

## Component Hierarchy & Directory Map

```
frontend/src/
├── main.tsx                             # Application entrypoint & React 19 root
├── App.tsx                              # AppShell: orchestrates routing, conversations, theme & modals
├── types.ts                             # Domain entities, raster metadata, verdicts & telemetry
├── api.ts                               # Client HTTP adapters for backend FastAPI services
├── styles.css                           # Liquid Glass design system, tokens, animations & responsive layout
│
├── components/
│   ├── navigation/
│   │   ├── TopNav.tsx                   # Centered floating glass pill + Right utility pill
│   │   ├── HistoryDrawer.tsx            # Slide-out conversation drawer (replaces permanent sidebar)
│   │   └── CommandPalette.tsx           # Modal command center (Ctrl/Cmd + K) for rapid switching
│   │
│   ├── composer/
│   │   ├── Composer.tsx                 # Floating pill composer with image attachment & Ctrl K badge
│   │   └── AttachmentTray.tsx           # Attached image chips with remove/swap controls
│   │
│   ├── chat/
│   │   ├── ConversationWorkspace.tsx    # Message stream container with auto-scroll anchor & header
│   │   ├── UserMessage.tsx              # Frosted glass card with user avatar & attached thumbnails
│   │   ├── AssistantMessage.tsx         # Structured glass card with intro, 2-col grid & follow-up chips
│   │   └── AnalysisProgress.tsx         # AI-style step-by-step progress indicator
│   │
│   ├── results/
│   │   ├── VisualResultCard.tsx         # Satellite comparison viewport with draggable < > slider & labels
│   │   ├── KeyFindingsCard.tsx          # 4-row structured metric card with icons & confidence badges
│   │   ├── ResultSummary.tsx            # Extended metric summary and verdict badges
│   │   ├── ReportCard.tsx               # Downloadable GeoProof PDF audit report card
│   │   └── CompareViewer.tsx            # Full-page comparison workbench
│   │
│   ├── inspectors/
│   │   ├── AnalysisInspector.tsx        # Technical telemetry & spectral trace drawer
│   │   ├── EvidenceDrawer.tsx           # Visual evidence gallery of all masks & spectral bands
│   │   ├── MapPanel.tsx                 # Interactive geospatial map with raster overlay & GeoJSON
│   │   ├── ImageLightbox.tsx            # Fullscreen high-resolution viewer with pan, zoom & download
│   │   └── ReportViewerModal.tsx        # Inline PDF report viewer modal with pagination
│   │
│   ├── views/
│   │   ├── HomeZeroState.tsx            # Landing hero: "Analyze Earth, Understand Change" + 5 chips
│   │   ├── CompareView.tsx              # Dual-image upload zone for temporal change / fusion
│   │   ├── ExploreGallery.tsx           # Historical satellite analysis gallery
│   │   ├── ReportsLibrary.tsx           # PDF audit document library
│   │   └── SettingsModal.tsx            # User preferences and appearance settings
│   │
│   ├── ui/
│   │   └── gradient-wave.tsx            # Harmonic multi-layer liquid gradient wave canvas component
│   │
│   └── shared/
│       ├── LiquidAuroraBackground.tsx   # Multi-layer background integrating GradientWave + SVG waves + particles
│       └── Toast.tsx                    # Non-intrusive notification toasts
```

---

## Key Component Specifications

### 1. Liquid Aurora Background (`LiquidAuroraBackground.tsx`)
- **Layers**:
  - `aurora-base-gradient`: Smooth base color (`#FAFBFF` light / `#080B12` dark).
  - `aurora-shapes-layer`: 6 distinct blurred radial gradient nodes drifting asynchronously.
  - `aurora-waves-layer`: 3 flowing liquid SVG sine/cosine curves drifting horizontally and vertically.
  - `aurora-particles-canvas`: Canvas with floating micro-particles simulating atmosphere.
- **Accessibility**: Automatically disables animations when `prefers-reduced-motion: reduce` is detected.

### 2. Top Navigation (`TopNav.tsx`)
- **Pill 1 (Center)**: Primary Navigation
  - Tabs: **Analyze** (active pill), **Compare**, **Explore**, **Reports**, separated by `topnav-tab-divider`.
- **Pill 2 (Right)**: Utilities
  - Search icon button (`search-palette-trigger`)
  - Vertical separator
  - Theme toggle button (`Sun` in light mode, `Moon` in dark mode)
  - Settings button
  - User avatar circle with `HS` initials and violet-blue gradient

### 3. Home Zero State (`HomeZeroState.tsx`)
- **Headline**: "Analyze Earth," with gradient span "Understand Change".
- **Subtitle**: "Ask questions, analyze satellite imagery, compare changes and generate insights — all in one place."
- **Composer**: Floating rounded pill with image upload button, textarea, `Ctrl K` hint, and gradient send button.
- **Chips**:
  - `[ Find land ]` (Emerald Green)
  - `[ Detect water ]` (Electric Blue)
  - `[ Count buildings ]` (Violet)
  - `[ Compare images ]` (Amber/Orange)
  - `[ Analyze change ]` (Pink/Coral)

### 4. Assistant Message & Results (`AssistantMessage.tsx` + `VisualResultCard.tsx` + `KeyFindingsCard.tsx`)
- **Card**: Translucent frosted glass surface (`rgba(255, 255, 255, 0.72)` light / `rgba(20, 25, 35, 0.62)` dark) with 26px radius.
- **Top**: Conversational introductory narrative.
- **Middle Grid**:
  - **Left**: `VisualResultCard` with interactive draggable `< >` divider, date pills (`Mar 2022`, `Mar 2024`), layer switcher, and action buttons.
  - **Right**: `KeyFindingsCard` with latency indicator and 4 structured rows (icon, label, value, confidence pill).
- **Bottom**: Concluding plain-language regional summary.
- **Follow-up Chips**:
  - `[ Show land cover ]`, `[ Count buildings ]`, `[ Explain confidence ]`, `[ Generate report ]`, `[ Analyze nearby area ]`.
