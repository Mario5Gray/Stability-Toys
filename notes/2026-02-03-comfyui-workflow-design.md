# ComfyUI Workflow Configuration System

## Data Model

### WorkflowConfig (Python dataclass)
```python
@dataclass
class WorkflowConfig:
    """Configuration for a single ComfyUI workflow."""
    name: str                          # Unique identifier (slug-like)
    display_name: str                  # Human-readable name
    description: str = ""              # Optional description
    workflow: Dict[str, Any] = None    # Either {filepath: "..."} or inline JSON
    # Override defaults when this workflow is selected:
    default_size: str = "512x512"
    default_steps: int = 20
    default_cfg: float = 7.0
    # Metadata
    tags: List[str] = field(default_factory=list)  # e.g., ["txt2img", "inpaint", "upscale"]


@dataclass
class WorkflowsYAML:
    """Root configuration from workflows.yml."""
    default_workflow: str              # Name of default workflow
    workflows: Dict[str, WorkflowConfig]
```

### YAML Structure (conf/workflows.yml)

**Primary format: File-based workflows (recommended)**
```yaml
default_workflow: lcm_cyberpony_xl

workflows:
  lcm_cyberpony_xl:
    display_name: "(Pony)CyberRealistic PonyXL + LCM-Sched + LCM Lora + DMD2"
    description: "Simple txt2img with KSampler"
    default_size: "1024x1024"
    default_steps: 7
    default_cfg: 0.0
    tags: ["img2img", "fast"]
    workflow:
      filepath: "/app/workflows/LCM_CYBERPONY_XL.json"  # <-- filepath MUST be first key

  papercut_xl:
    display_name: "(XL)Juggernaut with PaperCut XL Lora"
    description: "Lethargic model. gets it wrong all the time."
    default_size: "1024x1024"
    default_steps: 7
    default_cfg: 1.0
    tags: ["img2img", "papercut"]
    workflow:
      filepath: "/app/workflows/MAISIES_WORLD_LCM_IMG_LCM_PAPERCUT_XL.json"

  img2img-experiments:
    display_name: "Image to Image"
    description: "Denoise/refine existing image experimentation"
    default_size: "512x512"
    default_steps: 10
    default_cfg: 5.0
    tags: ["img2img", "denoise"]
    workflow:
      filepath: "/app/workflows/Tracking-LCM-DIFFS-API.json"
```

**Alternative: Inline JSON (still supported)**
```yaml
  inline-example:
    display_name: "Inline Workflow"
    description: "Workflow JSON stored directly in YAML"
    default_size: "512x512"
    default_steps: 20
    default_cfg: 7.0
    tags: ["inline"]
    workflow:
      "3":
        class_type: "KSampler"
        inputs:
          seed: 0
          # ... full ComfyUI workflow JSON
```

---

## API Structure

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/workflows` | List all workflows (summary) |
| GET | `/api/workflows/{name}` | Get single workflow (full detail + JSON) |
| POST | `/api/workflows/{name}` | Create or update workflow |
| PUT | `/api/workflows` | Bulk save all workflows |
| DELETE | `/api/workflows/{name}` | Delete workflow |
| POST | `/api/workflows/reload` | Reload from disk |

### Response Schemas

**GET /api/workflows** (list view - no content)
```json
{
  "default_workflow": "lcm_cyberpony_xl",
  "workflows": {
    "lcm_cyberpony_xl": {
      "display_name": "(Pony)CyberRealistic PonyXL + LCM-Sched",
      "description": "Simple txt2img with KSampler",
      "default_size": "1024x1024",
      "default_steps": 7,
      "default_cfg": 0.0,
      "tags": ["img2img", "fast"],
      "workflow": {
        "filepath": "/app/workflows/LCM_CYBERPONY_XL.json"
      }
    },
    "papercut_xl": {
      "display_name": "(XL)Juggernaut with PaperCut XL Lora",
      "description": "Lethargic model",
      "default_size": "1024x1024",
      "default_steps": 7,
      "default_cfg": 1.0,
      "tags": ["img2img", "papercut"],
      "workflow": {
        "filepath": "/app/workflows/MAISIES_WORLD_LCM_IMG_LCM_PAPERCUT_XL.json"
      }
    }
  }
}
```
*Note: `workflow` shows filepath reference but not loaded content*

**GET /api/workflows/{name}** (single workflow - includes loaded content)
```json
{
  "name": "lcm_cyberpony_xl",
  "display_name": "(Pony)CyberRealistic PonyXL + LCM-Sched",
  "description": "Simple txt2img with KSampler",
  "default_size": "1024x1024",
  "default_steps": 7,
  "default_cfg": 0.0,
  "tags": ["img2img", "fast"],
  "workflow": {
    "filepath": "/app/workflows/LCM_CYBERPONY_XL.json"
  },
  "workflow_content": {
    "3": { "class_type": "KSampler", "inputs": {...} },
    "4": { "class_type": "CheckpointLoaderSimple", "inputs": {...} }
  }
}
```
*Backend reads the file at `filepath` and returns as `workflow_content`*

**POST /api/workflows/{name}** (create/update)
```json
{
  "display_name": "My Custom Workflow",
  "description": "Optional description",
  "default_size": "768x768",
  "default_steps": 30,
  "default_cfg": 7.5,
  "tags": ["custom", "experimental"],
  "workflow": {
    "filepath": "/app/workflows/MY_CUSTOM_WORKFLOW.json"
  },
  "workflow_content": {
    "3": { "class_type": "KSampler", "inputs": {...} }
  }
}
```
*Backend should:*
1. *If `workflow.filepath` exists: write `workflow_content` to that file*
2. *If no filepath: store `workflow_content` inline in workflows.yml*
3. *Update workflows.yml with metadata*

**PUT /api/workflows** (bulk save + set default)
```json
{
  "default_workflow": "papercut_xl",
  "workflows": { ... }
}
```
*Used primarily for changing default_workflow*

---

## UI Component Design

### WorkflowEditor.jsx

Similar structure to ModeEditor but with workflow-specific features:

```
┌─────────────────────────────────────────────────────────────┐
│  Workflow Configuration                    [Reload] [+ Add] │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ ★  Text to Image (Basic)              [default]     │   │
│  │     Simple txt2img with KSampler                    │   │
│  │     Size: 512x512  Steps: 20  CFG: 7.0              │   │
│  │     Tags: txt2img, basic                            │   │
│  │                          [View JSON] [Edit] [Delete]│   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ ☆  Text to Image (LCM)                              │   │
│  │     Fast LCM sampling, 4-8 steps                    │   │
│  │     Size: 512x512  Steps: 4   CFG: 1.0              │   │
│  │     Tags: txt2img, lcm, fast                        │   │
│  │                          [View JSON] [Edit] [Delete]│   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Edit Form (expanded)

```
┌─────────────────────────────────────────────────────────────┐
│  Editing: txt2img-basic                                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Name (slug)        Display Name                            │
│  ┌──────────────┐   ┌──────────────────────────────────┐   │
│  │ txt2img-basic│   │ Text to Image (Basic)            │   │
│  └──────────────┘   └──────────────────────────────────┘   │
│                                                             │
│  Description                                                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Simple txt2img with KSampler                         │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  Default Size       Default Steps      Default CFG         │
│  ┌─────────────┐   ┌─────────────┐    ┌─────────────┐     │
│  │ 512x512   ▼│   │     20      │    │    7.0      │     │
│  └─────────────┘   └─────────────┘    └─────────────┘     │
│                                                             │
│  Tags (comma-separated)                                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ txt2img, basic                                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  Workflow JSON                              [Upload File]   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ {                                                    │  │
│  │   "3": {                                             │  │
│  │     "class_type": "KSampler",                        │  │
│  │     "inputs": { ... }                                │  │
│  │   }                                                  │  │
│  │ }                                                    │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│                                        [Cancel] [Save]      │
└─────────────────────────────────────────────────────────────┘
```

### Key UI Features

1. **Workflow JSON handling**:
   - Textarea for manual editing
   - "Upload File" button to load from .json file
   - JSON validation on save (parse test)
   - "View JSON" modal/drawer for read-only inspection

2. **Tags as chips**:
   - Display as badges
   - Edit as comma-separated input
   - Optional: filter/search by tag

3. **Default indicator**:
   - ★ filled star = current default
   - ☆ outline star = click to set as default

4. **Lazy loading**:
   - List view doesn't include full workflow JSON
   - Fetch full workflow only when editing or viewing

---

## File Structure

```
server/
  workflow_config.py      # WorkflowConfigManager (like mode_config.py)
  workflow_routes.py      # FastAPI router for /api/workflows/*

conf/
  workflows.yml           # Workflow definitions

lcm-sr-ui/src/components/config/
  WorkflowEditor.jsx      # Main component
  WorkflowForm.jsx        # Edit form (optional, can be inline)
  WorkflowJsonViewer.jsx  # Modal for viewing workflow JSON
```

---

## Integration Points

### When sending to ComfyUI:
```python
# In your ComfyUI client code
workflow_config = get_workflow_config()
workflow = workflow_config.get_workflow(current_workflow_name)

# Inject dynamic values into workflow JSON
workflow_json = inject_params(
    workflow.workflow,
    prompt=user_prompt,
    seed=seed,
    size=size or workflow.default_size,
    steps=steps or workflow.default_steps,
    cfg=cfg or workflow.default_cfg,
)

# Send to ComfyUI
comfy_client.queue_prompt(workflow_json)
```

### UI workflow selector:
Add a dropdown in the main generation UI to select active workflow (similar to mode selector if you have one).

---

## Migration Notes

If you have existing hardcoded workflows:
1. Export each as a named entry in workflows.yml
2. Update ComfyUI client to read from config instead of hardcoded JSON
3. Add workflow selector to UI

---

*Design complete. Mario handles backend, I'll build the UI component when ready.*
