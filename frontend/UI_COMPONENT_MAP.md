# SatQuery Orbit — Frontend Component Architecture Map

## Overview
The **SatQuery Orbit** frontend design replaces legacy permanent multi-sidebar GIS layouts with a conversational, spacious AI experience inspired by ChatGPT, Gemini, and Google Earth, engineered specifically for satellite imagery, GeoProof verification evidence, geospatial maps, and automated audit reports.

---

## Component Hierarchy & Directory Map

```
frontend/src/
├── main.tsx                         # Application entrypoint & React 19 root
├── App.tsx                          # AppShell: orchestrates router, history, drafts, themes & global drawers
├── types.ts                         # Strongly typed domain entities, UI states & GeoJSON models
├── api.ts                           # Client HTTP adapters for backend FastAPI services
├── styles.css                       # Centralized SatQuery Orbit design tokens & responsive styling
│
├── components/
│   ├── navigation/
│   │   ├── TopNav.tsx               # Floating Apple/ChatGPT-style nav with 4 core tabs
│   │   ├── HistoryDrawer.tsx        # Slide-over conversation drawer with search, pin, archive & rename
│   │   └── CommandPalette.tsx       # Modal command center (Ctrl/Cmd + K) for rapid switching
│   │
│   ├── composer/
│   │   ├── Composer.tsx             # Floating 22px rounded composer with dynamic placeholders & auto-resize
│   │   └── AttachmentTray.tsx       # Compact attachment cards with earlier/later badges & raster metadata
│   │
│   ├── chat/
│   │   ├── ConversationWorkspace.tsx# Centered (900-1000px) message stream with bottom anchor
│   │   ├── UserMessage.tsx          # Right-aligned compact bubble with attached thumbnail chips
│   │   ├── AssistantMessage.tsx     # Unboxed conversational natural language response + structured results
│   │   └── AnalysisProgress.tsx     # AI-style step-by-step progress card (collapses upon completion)
│   │
│   ├── results/
│   │   ├── ResultSummary.tsx        # 3-4 primary clear metrics with icons & verified status pills
│   │   ├── VisualResultCard.tsx     # Large 16:9/natural ratio imagery with layer switcher & swipe compare
│   │   ├── ReportCard.tsx           # Modern file-attachment style card for GeoProof PDF report
│   │   └── CompareViewer.tsx        # Before/After comparison view with draggable divider & overlay
│   │
│   ├── inspectors/
│   │   ├── AnalysisInspector.tsx    # Slide-out drawer with 4 tabs: Overview, Evidence, Geospatial, Technical
│   │   ├── EvidenceDrawer.tsx       # Visual evidence gallery of all masks, spectral outputs & disagreements
│   │   ├── MapPanel.tsx             # Interactive geospatial map with raster overlay, vector polygons & HUD
│   │   ├── ImageLightbox.tsx        # Fullscreen high-resolution viewer with pan, wheel-zoom, fit & download
│   │   └── ReportViewerModal.tsx    # Inline PDF report viewer with pagination, download & tab controls
│   │
│   ├── views/
│   │   ├── HomeZeroState.tsx        # "Ask the Earth anything", authentic Earth visual, composer & recent cards
│   │   ├── CompareView.tsx          # Dual-image upload zone for temporal change or optical/SAR fusion
│   │   ├── ExploreGallery.tsx       # Visual memory gallery of all previous satellite analysis runs
│   │   └── ReportsLibrary.tsx       # Document library with grid/list view, date grouping & filters
│   │
│   └── shared/
│       └── Toast.tsx                # Non-intrusive toast notifications for feedback events
```

---

## Detailed Component Specifications

### 1. AppShell (`App.tsx`)
- **Responsibilities**:
  - Handles client-side URL hash routing (`#/analyze`, `#/chat/:id`, `#/compare`, `#/explore`, `#/reports`).
  - Manages active conversation and synchronizes optimistic user queries with `/api/v1/chats/:id/messages`.
  - Maintains local preferences: Dark/Light theme, pinned chat IDs, archived chat IDs, draft messages.
  - Controls drawer and modal visibility (History, Inspector, Evidence, Map, Lightbox, Report, Settings, Command Palette).

### 2. Top Navigation (`TopNav.tsx`)
- **Props**:
  - `activeTab`: `"analyze" | "compare" | "explore" | "reports"`
  - `onSelectTab`: `(tab: NavTab) => void`
  - `isHistoryOpen`: `boolean`
  - `onToggleHistory`: `() => void`
  - `theme`: `"dark" | "light"`
  - `onToggleTheme`: `() => void`
  - `onOpenCommandPalette`: `() => void`
  - `onOpenSettings`: `() => void`
- **Features**:
  - 60px height with backdrop blur (`backdrop-filter: blur(14px)`).
  - Left: History drawer toggle + SatQuery Orbit logo (returns to start screen).
  - Center: Exactly 4 primary desktop tabs: **Analyze**, **Compare**, **Explore**, **Reports**.
  - Right: Search trigger (`Ctrl+K`), Theme toggle, Settings shortcut, User profile badge.

### 3. History Drawer (`HistoryDrawer.tsx`)
- **Props**:
  - `isOpen`: `boolean`
  - `chats`: `ChatSummary[]`
  - `currentChatId`: `string | null`
  - `pinnedChatIds`: `string[]`, `archivedChatIds`: `string[]`
  - Handlers: select, new, rename, delete, toggle pin, toggle archive.
- **Features**:
  - 320px width desktop slide-over (full-height on mobile).
  - Date sections: Pinned, Today, Yesterday, Previous 7 Days, Previous 30 Days, Older.
  - Hover overflow menu: Rename, Pin/Unpin, Archive, Delete with confirmation modal.
  - Fast client-side search across titles and preview text.

### 4. Floating Composer (`Composer.tsx`)
- **Props**:
  - `onSendMessage`: `(query: string, pairType: string, files: File[]) => Promise<void>`
  - `busy`: `boolean`
  - `activeContextImageName`: `string | null`
  - `showSuggestions`: `boolean`
- **Features**:
  - 22px border radius, max-width 880px, auto-resizing textarea (min 54px up to 180px).
  - Drag-and-drop file dropzone, file picker, clipboard paste support.
  - Suggestion chips: "Find water", "Count buildings", "Show land cover", "Compare images".
  - Multi-image support with "Earlier/Later" or "Optical/SAR" tagging and swap control.
  - Context continuity tray ("Using: coastal_scene.tif") with detachment action.

### 5. Visual Result Card (`VisualResultCard.tsx`)
- **Props**:
  - `result`: `AnalysisResponse`
  - `onOpenLightbox`: `(imageUrl: string, title: string) => void`
  - `onOpenMap`: `(result: AnalysisResponse) => void`
- **Features**:
  - 16:9 ratio high-resolution satellite imagery preview.
  - Dynamic layer switcher (Original, Result Overlay, and actual generated masks).
  - Interactive draggable swipe divider for before/after or optical/SAR comparison.
  - Actions: Fullscreen Lightbox expand, Interactive Map inspection, Unified Download menu (PNG, GeoJSON, PDF).

### 6. Result Summary (`ResultSummary.tsx`)
- **Props**:
  - `result`: `AnalysisResponse`
- **Features**:
  - Strict maximum of 3-4 primary clear numbers.
  - Formats areas dynamically (`m²`, `ha`, `km²`).
  - Accessible status badge: Supported (green), Supported with limitations (amber), Disputed (red), Inconclusive (muted).

### 7. Analysis Inspector (`AnalysisInspector.tsx`)
- **Props**:
  - `isOpen`: `boolean`, `result`: `AnalysisResponse | null`
  - `onClose`: `() => void`, `onOpenImage`: `(url: string, title: string) => void`
- **Features**:
  - Slide-out drawer organizing advanced telemetry into 4 tabs:
    1. **Overview**: Query intent, mode, calibrated confidence breakdown, documented limitations.
    2. **Evidence**: Multi-witness artifacts, individual confidence scores, metrics dictionary.
    3. **Geospatial**: Coordinate Reference System (CRS), bounds, pixel resolution, raster dimensions.
    4. **Technical**: Neural backbones executed, latency timings, execution trace steps.

### 8. Interactive Map Panel (`MapPanel.tsx`)
- **Props**:
  - `isOpen`: `boolean`, `result`: `AnalysisResponse | null`
  - `onClose`: `() => void`
- **Features**:
  - 50vw split-screen on desktop (fullscreen toggle available), full-height on mobile.
  - Geospatial grid coordinate HUD with raster overlay and vector polygons.
  - Feature click inspection: area, location, copy coordinates.
