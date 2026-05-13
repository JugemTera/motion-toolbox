# Embedding Motion Visualizations in Web Applications

This directory contains examples for embedding motion visualizations in various web frameworks using the unified visualizer's `to_scene_data()` method.

## Overview

The unified visualizer provides two main approaches for web integration:

### 1. **Standalone HTML Files** (Simple)
Generate complete HTML files with embedded visualizations:

```python
from motion_toolbox.visualization import Visualizer

vis = Visualizer()
vis.visualize(motion_data, "output.html", mode='skeleton')
```

Best for:
- Quick prototypes
- Standalone demos
- Sharing visualizations as single files

### 2. **Scene Data Export** (Flexible)
Export motion data as JSON for custom frontend integration:

```python
from motion_toolbox.visualization import Visualizer

vis = Visualizer()
scene_data = vis.to_scene_data(motion_data, mode='skeleton')

# scene_data is a JSON-serializable dict:
# {
#     'type': 'skeleton',
#     'positions': [...],  # Joint positions
#     'kinematic_chains': [...],  # Bone connections
#     'chain_colors': [...],  # Colors
#     'fps': 20,
#     'num_frames': 120
# }

import json
json_str = json.dumps(scene_data)  # Send to frontend
```

Best for:
- Web applications (Flask, FastAPI, Django)
- Single-page applications (React, Vue, Angular)
- Custom Three.js visualizations
- User studies with custom UI

## Web Framework Examples

### Flask

See [`flask_embedding_example.py`](flask_embedding_example.py) for a complete example.

```python
from flask import Flask, jsonify
from motion_toolbox.visualization import Visualizer

app = Flask(__name__)

@app.route('/api/motion/<motion_id>')
def get_motion(motion_id):
    motion_data = load_motion_from_db(motion_id)
    vis = Visualizer()
    scene_data = vis.to_scene_data(motion_data, mode='skeleton')
    return jsonify(scene_data)
```

### FastAPI

```python
from fastapi import FastAPI
from motion_toolbox.visualization import Visualizer

app = FastAPI()

@app.get("/motion/{motion_id}")
async def get_motion(motion_id: str):
    motion_data = await load_motion(motion_id)
    vis = Visualizer()
    scene_data = vis.to_scene_data(motion_data, mode='skeleton')
    return scene_data  # FastAPI auto-serializes to JSON
```

### Django

```python
from django.http import JsonResponse
from motion_toolbox.visualization import Visualizer

def motion_view(request, motion_id):
    motion_data = Motion.objects.get(id=motion_id).get_data()
    vis = Visualizer()
    scene_data = vis.to_scene_data(motion_data, mode='skeleton')
    return JsonResponse(scene_data)
```

## Frontend Integration

### React Component

```jsx
import React, { useEffect, useState } from 'react';
import * as THREE from 'three';

function MotionViewer({ motionId }) {
  const [sceneData, setSceneData] = useState(null);
  
  useEffect(() => {
    fetch(`/api/motion/${motionId}`)
      .then(res => res.json())
      .then(data => setSceneData(data));
  }, [motionId]);
  
  useEffect(() => {
    if (!sceneData) return;
    
    // Create Three.js scene using sceneData
    const scene = new THREE.Scene();
    // ... render skeleton using sceneData.positions, 
    // sceneData.kinematic_chains, etc.
  }, [sceneData]);
  
  return <div ref={containerRef} />;
}
```

### Vue Component

```vue
<template>
  <div ref="viewer"></div>
</template>

<script>
import * as THREE from 'three';

export default {
  props: ['motionId'],
  
  async mounted() {
    const response = await fetch(`/api/motion/${this.motionId}`);
    const sceneData = await response.json();
    
    // Create Three.js visualization
    this.renderScene(sceneData);
  },
  
  methods: {
    renderScene(sceneData) {
      // Use sceneData to create Three.js scene
    }
  }
}
</script>
```

## Scene Data Format

### Skeleton Mode

```json
{
  "type": "skeleton",
  "title": "Motion Animation",
  "fps": 20,
  "num_frames": 120,
  "num_joints": 22,
  "positions": [
    [[x, y, z], ...],  // Frame 0: 22 joints
    [[x, y, z], ...],  // Frame 1: 22 joints
    ...
  ],
  "kinematic_chains": [
    [0, 2, 5, 8, 11],  // Right leg
    [0, 1, 4, 7, 10],  // Left leg
    ...
  ],
  "chain_colors": [
    "#ff6b6b",  // Red for right leg
    "#4ecdc4",  // Cyan for left leg
    ...
  ],
  "metadata": {
    "format": "motion_toolbox_scene_v1",
    "three_js_compatible": true
  }
}
```

### Mesh Mode

For mesh visualizations, it's recommended to:
1. Generate GLB file with `Visualizer.visualize()`
2. Serve the GLB file statically
3. Reference it in your frontend

```python
# Backend: Generate GLB
vis.visualize(motion_data, "static/motions/motion_123.glb", mode='mesh')

# Frontend: Load GLB with model-viewer or Three.js GLTFLoader
```

## Best Practices

1. **Caching**: Cache generated scene data for frequently accessed motions
2. **Compression**: Use gzip compression for large position arrays
3. **Streaming**: For very long sequences, consider streaming frame data
4. **Progressive Loading**: Load skeleton first, then mesh details
5. **Error Handling**: Always handle format detection and conversion errors

## Security Considerations

- Validate motion IDs before database queries
- Sanitize user-provided motion data
- Rate limit API endpoints
- Consider authentication for sensitive motion data

## Performance Tips

- **Client-side rendering**: Let Three.js handle animation on the client
- **Data decimation**: Downsample motion for preview, full quality on demand
- **WebGL**: Ensure browser supports WebGL for Three.js
- **Mobile**: Test on mobile devices, adjust visualization complexity
