---
tags:
- geospatial
- geobase
- building-detection
- building-footprint-detection
license: mit
---
| <img src="https://upload.wikimedia.org/wikipedia/commons/6/6a/JavaScript-logo.png" width="28" height="28"> | [GeoAi](https://www.npmjs.com/package/geoai) |
|---|---|



> `task = building-detection`

### 🛠 Model Purpose
This model is part of the **[GeoAi](https://github.com/decision-labs/geoai.js)** javascript library.

**GeoAi** enables geospatial AI inference **directly in the browser or Node.js** without requiring a heavy backend.

**GeoAi** pipeline accepts **geospatial polygons** as input (in GeoJSON format) and outputs results as a **GeoJSON FeatureCollection**, ready for use with libraries like **Leaflet** and **Mapbox GL**.

<video controls autoplay loop width="1024" height="720" src="https://geobase-docs.s3.amazonaws.com/geobase-ai-assets/building-detection.mp4"></video>

---
### 🚀 Demo

Explore the model in action with the interactive [Demo](https://docs.geobase.app/geoai-live/tasks/building-detection).

### 📦 Model Information
- **Architecture**: MaskRCNN
- **Source Model**: https://opengeoai.org/examples/train_building_footprints_usa
- **Quantization**: Yes
---

### 💡 Example Usage

```javascript
import { geoai } from "geoai";

// Example polygon (GeoJSON)
const polygon = {
  type: "Feature",
  properties: {},
  geometry: {
    coordinates: [
      [
        [-117.59239617156095, 47.653614113446906],
        [-117.59239617156095, 47.652878388765174],
        [-117.59040545822742, 47.652878388765174],
        [-117.59040545822742, 47.653614113446906],
        [-117.59239617156095, 47.653614113446906]
      ],
    ],
    type: "Polygon",
  },
} as GeoJSON.Feature;

// Initialize pipeline
const pipeline = await geoai.pipeline(
  [{ task: "building-detection" }],
  providerParams
);
 
// Run detection
const result = await pipeline.inference({
  inputs: { polygon }
});
 
// Sample output format
// {
//     "detections": {
//         "type": "FeatureCollection",
//         "features": [
//             {
//                 "type": "Feature",
//                 "properties": {
//                 },
//                 "geometry": {
//                     "type": "Polygon",
//                     "coordinates": [
//                         [
//                             [54.69479163045772, 24.766579711184693],
//                             [54.69521093930892, 24.766579711184693],
//                             [54.69521093930892, 24.766203991224682],
//                             [54.69479163045772, 24.766203991224682],
//                             [54.69479163045772, 24.766579711184693],
//                         ]
//                     ]
//                 }
//             },
//             {"type": 'Feature', "properties": {…}, "geometry": {…}}, 
//             {"type": 'Feature', "properties": {…}, "geometry": {…}},
//         ]
//     },
//     "geoRawImage": GeoRawImage {data: Uint8ClampedArray(1048576), width: 512, height: 512, channels: 4, bounds: {…}, …}
// }

```
### 📖 Documentation & Demo

- GeoBase Docs: https://docs.geobase.app/geoai
- NPM Package: https://www.npmjs.com/package/geoai
- Demo Playground: https://docs.geobase.app/geoai-live/tasks/building-detection
- GitHub Repo: https://github.com/decision-labs/geoai.js