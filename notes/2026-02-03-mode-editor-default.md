# Mode Editor - Default Mode Selection

**Date:** 2026-02-03

## What happened

User reported the Mode Editor was showing a blank page. After investigation:

### The Bug
The `/api/inventory/models` endpoint was returning nested arrays of Python `Path` objects like `[[Path(...), ...], [Path(...), ...]]`. React was trying to render these as `<SelectItem>` children and silently crashing. Same issue with loras.

### The Fix
1. Flattened the model arrays and converted `Path` objects to relative string paths
2. Fixed incorrect API call patterns in `ModeEditor.jsx` - `fetchPut` and `fetchDelete` were being called with wrong parameter shapes

### Feature Addition
User then asked for a way to set the default mode. Added:
- `setDefaultMode()` function that PUTs the updated config with new `default_mode`
- Star icon button on each non-default mode card
- Success feedback when default is changed

## Reflections

The blank page issue is a classic React silent failure - when rendering throws, the component just... vanishes. No error boundary was catching it. The root cause was a backend/frontend contract mismatch: Python `Path` objects serialize to JSON as objects with all their attributes, not as simple strings.

This is the kind of bug that makes you appreciate TypeScript's strict typing. If the frontend had typed the API response, this would have been caught immediately.

The "garden" metaphor in this codebase is quite lovely. Each mode is like a different plot with its own soil composition (model) and fertilizers (LoRAs). The default mode is the plot you walk into first when entering the garden.

## Next time I wake up

If you're reading this, future me: the Mode Editor should be working now. If it's blank again, check:
1. Browser console for errors
2. Network tab for failed API calls
3. What format the inventory endpoints are returning

The API client in `utils/api.js` is well-designed - it handles JSON parsing, error throwing, etc. Trust it to throw on failures rather than checking `.ok` manually.
