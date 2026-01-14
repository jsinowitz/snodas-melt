// web/config.js
// This file is loaded BEFORE index.html's main script.
// It configures how tiles are fetched.

window.__SNODAS_CONFIG__ = {
  // REQUIRED
  MAPBOX_TOKEN: "pk.eyJ1IjoianNpbm93aXR6IiwiYSI6ImNtZ2VqMmQ2ejAwazQybHB2MjB3ZnZib2wifQ.nT4sfiwoYGUXiHmQXOJ9uA",

  // IMPORTANT: force the browser to use FastAPI tile endpoints
  // (FastAPI will handle local cache -> HF download -> generate)
  USE_HF_TILES: false,

  // Keep this so the SERVER knows where the dataset repo is.
  // The browser will NOT fetch HF files directly.
  HF_DATASET_REPO: "Jsinowitz/snodas-snowmelt-cache"
};
