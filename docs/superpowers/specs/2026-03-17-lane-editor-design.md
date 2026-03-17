# Lane Editor Design

**Date:** 2026-03-17
**Status:** Approved
**Scope:** Standalone interactive lane polygon editor using OpenCV, saving back to YAML config

## Problem

Users need to visually verify and adjust lane polygons on their video feed. Currently, lane coordinates must be manually edited in the YAML config file, which is tedious and error-prone without visual feedback.

## Requirements

- Display the first frame of a video with lane polygons overlaid
- Move existing polygon vertices by click-and-drag
- Insert new vertices on polygon edges
- Delete vertices (minimum 3 enforced)
- Draw entirely new lane polygons
- Delete entire lanes
- Save changes back to the original YAML config file
- Standalone CLI tool: `traffic-lane-editor`

## Non-Requirements

- Undo/redo
- Multi-select or bulk operations
- Editing non-lane config fields
- Real-time detection while editing

## Design

### LaneEditor Class

A new module `traffic_detection_kpi/lane_editor.py` with a `LaneEditor` class.

**Interface:**

```python
class LaneEditor:
    def __init__(self, frame: np.ndarray, lanes: list[dict], lane_colors: list[tuple]) -> None:
        """Initialize with background frame and lane data.

        Args:
            frame: First video frame (numpy array)
            lanes: List of {"name": str, "polygon": list[list[int]]} dicts
            lane_colors: BGR color tuples, one per lane
        """
        ...

    def run(self) -> tuple[bool, list[dict]]:
        """Run the editor event loop.

        Returns:
            (modified, lanes) — whether changes were made, and the
            current lane data (original or edited).
        """
        ...
```

### Display

- First video frame as a static background
- Lane polygons drawn with semi-transparent colored fill + solid border (reuses annotator color palette)
- Vertices shown as small circles:
  - Hollow white circle (radius 6) for unselected vertices
  - Filled colored circle (radius 8) for the selected vertex
- Lane names drawn at polygon centroid in the lane's color
- Status bar (dark strip at bottom, 40px) showing:
  - Current mode: "EDIT" or "DRAW (click to place points, Enter to finish)"
  - Instructions: "n: new lane | d: delete lane | q: quit | Esc: cancel"

### Mouse Interactions

All interactions use `cv2.setMouseCallback`:

- **Left click near a vertex** (within 15px) — select vertex, enter drag mode
- **Left mouse drag** — reposition selected vertex
- **Left mouse release** — drop vertex at new position
- **Left click on a polygon edge** (within 10px, not near a vertex) — insert a new vertex at the closest point on that edge segment, rounded to integer coordinates
- **Left click on empty space** — deselect current selection
- **Right click on a vertex** — delete it (only if polygon has > 3 vertices; otherwise ignore)
- **Right click inside a polygon** — select that lane as the active lane (for deletion with `d`)

**Coordinate handling:**
- All vertex coordinates are stored as `int`. Mouse event coordinates from OpenCV are already `int`. Edge insertion uses point-to-segment projection which may produce floats — these must be rounded to `int` before storage.

**Hit detection math:**
- Vertex proximity: Euclidean distance from click point to each vertex, threshold 15px
- Edge proximity: point-to-line-segment distance for each consecutive pair of vertices, threshold 10px
- Vertex check takes priority over edge check (prevents accidental inserts when dragging)

### Keyboard Shortcuts

- `n` — enter "new lane" draw mode:
  - Left clicks place vertices (shown as connected lines)
  - `Enter` finishes the polygon (minimum 3 points required, otherwise ignored)
  - Prompts for lane name in terminal via `input()`
  - `Esc` cancels and returns to edit mode
- `d` — delete the currently selected lane (the one last right-clicked inside). Vertex drag-selection does NOT update the active-lane-for-deletion state — only right-click inside a polygon sets it. If no lane has been right-click-selected, `d` does nothing
- `Esc` — cancel current draw mode, deselect all
- `q` — quit the editor. If currently in draw mode, first cancels draw mode (discards in-progress polygon), then proceeds with quit:
  - If changes were made, prompt in terminal: "Save changes? (y/n/c): "
  - `y` — return `(True, updated_lanes)`
  - `n` — return `(False, original_lanes)`
  - `c` — cancel quit, return to editor

### Draw Mode

When `n` is pressed:
- Status bar changes to "DRAW — click to place points, Enter to finish, Esc to cancel"
- Left clicks add vertices, shown as circles connected by lines
- The in-progress polygon is drawn with dashed lines
- `Enter` completes the polygon if >= 3 points, prompts for name
- New lane gets the next color from the palette

### Editor CLI Entry Point

A new module `traffic_detection_kpi/editor_cli.py`:

```python
def main():
    parser = argparse.ArgumentParser(
        description="Interactive lane polygon editor"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--video", help="Path to video file (overrides config video_path)")
    args = parser.parse_args()
```

**Flow:**
1. Load config via `load_config(args.config)`
2. Determine video path: `args.video or config.video_path`
3. Open video with `cv2.VideoCapture`, read first frame, release
4. Build lane data list from config: `[{"name": l.name, "polygon": l.polygon} for l in config.lanes]`
5. Compute lane colors: `colors = [LANE_PALETTE[i % len(LANE_PALETTE)] for i in range(len(config.lanes))]` using the public `LANE_PALETTE` constant from `annotator.py` (rename `_PALETTE` to `LANE_PALETTE` to make it importable)
6. Launch `LaneEditor(frame, lanes, colors).run()`
7. If `modified` is True, call `save_lanes_to_config(args.config, updated_lanes)`

### Config Writing

A function `save_lanes_to_config(config_path: str, lanes: list[dict])` in `editor_cli.py`:

1. Read the original YAML file with `yaml.safe_load`
2. Replace the `lanes` key with the updated lane data
3. Write back with `yaml.safe_dump(..., default_flow_style=False, sort_keys=False)`
4. This preserves all non-lane config field values and key ordering. **Note:** YAML comments in the original file will be lost — this is a known limitation of `yaml.safe_dump`

### pyproject.toml Change

Add a new script entry point:

```toml
[project.scripts]
traffic-kpi = "traffic_detection_kpi.__main__:main"
traffic-lane-editor = "traffic_detection_kpi.editor_cli:main"
```

### Files Changed

| File | Change |
|------|--------|
| `traffic_detection_kpi/lane_editor.py` | **New** — LaneEditor class |
| `traffic_detection_kpi/editor_cli.py` | **New** — CLI entry point + config save function |
| `traffic_detection_kpi/annotator.py` | Rename `_PALETTE` to `LANE_PALETTE` (public) |
| `pyproject.toml` | Add `traffic-lane-editor` script entry point |
| `tests/test_lane_editor.py` | **New** — unit tests |

### Testing

**New tests in `tests/test_lane_editor.py`:**

- `test_save_lanes_to_config_roundtrip`: load config, modify lanes, save, reload, verify lanes match and non-lane fields preserved
- `test_save_lanes_preserves_other_config`: save lanes to a config with model/tracker/output_dir, verify those fields are unchanged
- `test_vertex_hit_detection`: verify point-near-vertex math returns correct vertex index within threshold
- `test_edge_hit_detection`: verify point-near-edge math returns correct edge index within threshold
- `test_no_delete_below_three_vertices`: verify a 3-vertex polygon cannot have vertices deleted

**Not unit tested:** Visual rendering, mouse drag behavior, keyboard shortcuts (these require a live OpenCV window).
