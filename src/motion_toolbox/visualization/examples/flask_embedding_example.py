"""Example: Embedding motion visualization in Flask application.

This example shows how to use the unified visualizer's to_scene_data()
method to export motion data for display in a Flask web application.
"""

import numpy as np
from flask import Flask, jsonify, render_template

from motion_toolbox.visualization import Visualizer

app = Flask(__name__)


@app.route("/")
def index():
    """Render main page with visualization."""
    return render_template("motion_viewer.html")


@app.route("/api/motion_data")
def get_motion_data():
    """API endpoint that returns motion scene data as JSON."""
    # Load or generate motion data
    # For this example, we'll use dummy data
    motion_data = np.random.randn(120, 263)  # HumanML3D format

    # Create visualizer and export scene data
    vis = Visualizer()
    scene_data = vis.to_scene_data(motion_data, mode="skeleton", fps=20, title="Motion from Database")

    # Return as JSON (Flask will serialize the dict)
    return jsonify(scene_data)


# HTML template (save as templates/motion_viewer.html):
TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Motion Viewer</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head>
<body>
    <div id="viewer" style="width: 100vw; height: 100vh;"></div>
    
    <script>
        // Fetch motion data from API
        fetch('/api/motion_data')
            .then(response => response.json())
            .then(sceneData => {
                // sceneData contains:
                // - positions: joint positions array
                // - kinematic_chains: bone connections
                // - chain_colors: colors for bones
                // - fps, num_frames, etc.
                
                // Use sceneData to create Three.js visualization
                console.log('Motion data loaded:', sceneData);
                // ... implement Three.js rendering ...
            });
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(debug=True)
