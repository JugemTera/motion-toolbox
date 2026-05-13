"""Separate viewers grid for GLB motion animations.

This module provides functionality to display multiple motion animations
in separate HTML viewer instances arranged in a grid layout.
Each animation has its own model-viewer instance with shared playback controls.
"""

import base64
from pathlib import Path


class SeparateViewersGrid:
    """Creates standalone HTML with multiple separate viewer instances in a grid layout.

    This class provides interactive 3D motion visualization by embedding multiple GLB animations
    as separate model-viewer instances in a single HTML file with synchronized playback controls.
    """

    def __init__(self):
        """Initialize the multi-animation viewer generator."""
        pass

    def create_grid_viewer(
        self,
        glb_paths: list[str],
        output_path: str,
        title: str = "Multi-Animation Grid Viewer",
        labels: list[str] | None = None,
        grid_cols: int | None = None,
    ) -> None:
        """Create a standalone HTML file with multiple GLB animations in a grid.

        Args:
            glb_paths (List[str]): List of paths to GLB files to embed
            output_path (str): Path where the HTML file will be saved
            title (str): Title for the HTML page
            labels (Optional[List[str]]): Optional labels for each animation. If None, no labels shown.
            grid_cols (Optional[int]): Number of columns in grid. If None, auto-calculated based on count.
        """
        if not glb_paths:
            raise ValueError("glb_paths must contain at least one GLB file")

        # Auto-calculate grid columns if not specified
        num_animations = len(glb_paths)
        if grid_cols is None:
            # Create a roughly square grid
            grid_cols = int(num_animations**0.5) + (1 if num_animations**0.5 % 1 > 0.5 else 0)
            grid_cols = max(1, min(grid_cols, 4))  # Clamp between 1-4 columns

        # Encode all GLB files
        glb_data_list = [self._encode_glb_to_base64(path) for path in glb_paths]

        # Generate labels if not provided
        if labels is None:
            labels = []  # Empty list means no labels
        elif len(labels) != len(glb_paths):
            raise ValueError(f"Number of labels ({len(labels)}) must match number of GLB files ({len(glb_paths)})")

        # Generate the complete HTML
        html_content = self._generate_grid_html_template(
            glb_data_list=glb_data_list,
            title=title,
            labels=labels,
            grid_cols=grid_cols,
        )

        # Write HTML file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"✅ Multi-animation grid viewer created: {output_path}")
        print(f"   {len(glb_paths)} animations in {grid_cols}-column grid")

    def _encode_glb_to_base64(self, glb_path: str) -> str:
        """Encode GLB file to base64 string.

        Args:
            glb_path (str): Path to the GLB file

        Returns:
            str: Base64 encoded GLB data with data URL prefix
        """
        glb_file = Path(glb_path)
        if not glb_file.exists():
            raise FileNotFoundError(f"GLB file not found: {glb_path}")

        with open(glb_file, "rb") as f:
            glb_data = f.read()

        # Encode as base64 with proper data URL format
        base64_data = base64.b64encode(glb_data).decode("utf-8")
        return f"data:model/gltf-binary;base64,{base64_data}"

    def _generate_grid_html_template(
        self,
        glb_data_list: list[str],
        title: str,
        labels: list[str],
        grid_cols: int,
    ) -> str:
        """Generate the complete HTML template with multiple embedded GLBs in a grid.

        Args:
            glb_data_list (List[str]): List of base64 encoded GLB data URLs
            title (str): Page title
            labels (List[str]): Labels for each animation
            grid_cols (int): Number of columns in the grid

        Returns:
            str: Complete HTML content
        """
        num_animations = len(glb_data_list)
        show_labels = len(labels) == num_animations

        # Generate model-viewer elements
        viewer_elements = []
        for idx, glb_data in enumerate(glb_data_list):
            label_html = ""
            if show_labels:
                label_html = f'<div class="animation-label">{labels[idx]}</div>'

            viewer_html = f'''
                <div class="grid-item">
                    {label_html}
                    <model-viewer
                        id="viewer{idx}"
                        class="grid-viewer"
                        src="{glb_data}"
                        camera-controls
                        camera-orbit="-180deg 75deg 8m"
                        camera-target="0m 1m 0m"
                        min-camera-orbit="auto auto 4m"
                        max-camera-orbit="auto auto 20m"
                        touch-action="pan-y"
                        shadow-intensity="0.5"
                        shadow-softness="1"
                        interaction-prompt="none"
                        autoplay
                    ></model-viewer>
                </div>'''
            viewer_elements.append(viewer_html)

        viewers_html = "\n".join(viewer_elements)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    
    <!-- Model Viewer Script -->
    <script
        type="module"
        src="https://ajax.googleapis.com/ajax/libs/model-viewer/4.0.0/model-viewer.min.js"
    ></script>

    <style>
        * {{
            box-sizing: border-box;
        }}

        body {{
            margin: 0;
            padding: 20px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: #f5f5f5;
        }}

        .container {{
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }}

        .header {{
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }}

        .header h1 {{
            margin: 0 0 8px 0;
            font-size: 24px;
            font-weight: 600;
        }}

        .header-info {{
            margin: 0;
            font-size: 14px;
            opacity: 0.9;
        }}

        .grid-container {{
            display: grid;
            grid-template-columns: repeat({grid_cols}, 1fr);
            gap: 16px;
            padding: 16px;
            background: #fafafa;
        }}

        .grid-item {{
            position: relative;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            min-height: 400px;
            display: flex;
            flex-direction: column;
        }}

        .animation-label {{
            padding: 12px;
            background: #f8fafc;
            border-bottom: 1px solid #e5e7eb;
            font-size: 13px;
            font-weight: 500;
            color: #374151;
            text-align: center;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}

        .grid-viewer {{
            width: 100%;
            height: 100%;
            min-height: 350px;
            flex: 1;
        }}

        .controls-container {{
            position: sticky;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(0, 0, 0, 0.85);
            backdrop-filter: blur(10px);
            border-top: 1px solid rgba(255, 255, 255, 0.1);
            padding: 16px 20px;
            z-index: 1000;
        }}

        .controls-wrapper {{
            max-width: 1600px;
            margin: 0 auto;
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .control-button {{
            color: #ffffff;
            background: rgba(255, 255, 255, 0.15);
            border: none;
            padding: 10px;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 6px;
            backdrop-filter: blur(4px);
            min-width: 40px;
        }}

        .control-button:hover {{
            background: rgba(255, 255, 255, 0.25);
            transform: scale(1.05);
        }}

        .control-button:active {{
            transform: scale(0.98);
        }}

        .control-icon {{
            width: 20px;
            height: 20px;
        }}

        .progress-slider {{
            flex: 1;
            height: 6px;
            margin: 0 12px;
            background: rgba(255, 255, 255, 0.3);
            border-radius: 3px;
            cursor: pointer;
            -webkit-appearance: none;
            appearance: none;
            outline: none;
        }}

        .progress-slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            appearance: none;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: #ffffff;
            cursor: pointer;
            border: 2px solid #667eea;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
            transition: transform 0.2s ease;
        }}

        .progress-slider::-webkit-slider-thumb:hover {{
            transform: scale(1.15);
        }}

        .progress-slider::-moz-range-thumb {{
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: #ffffff;
            cursor: pointer;
            border: 2px solid #667eea;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.4);
        }}

        .time-display {{
            color: white;
            font-size: 14px;
            font-variant-numeric: tabular-nums;
            min-width: 100px;
            text-align: center;
        }}

        .info-panel {{
            padding: 16px 20px;
            background: #f8fafc;
            border-top: 1px solid #e5e7eb;
        }}

        .info-text {{
            margin: 0;
            color: #6b7280;
            font-size: 13px;
            text-align: center;
        }}

        /* Loading animation */
        .loading-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(255, 255, 255, 0.95);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 10;
        }}

        .spinner {{
            width: 36px;
            height: 36px;
            border: 3px solid #e5e7eb;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }}

        .loading-text {{
            margin-top: 12px;
            color: #6b7280;
            font-size: 12px;
        }}

        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}

        /* Responsive design */
        @media (max-width: 1200px) {{
            .grid-container {{
                grid-template-columns: repeat({max(1, grid_cols - 1)}, 1fr);
            }}
        }}

        @media (max-width: 768px) {{
            body {{
                padding: 10px;
            }}

            .grid-container {{
                grid-template-columns: 1fr;
                gap: 12px;
                padding: 12px;
            }}

            .grid-item {{
                min-height: 300px;
            }}

            .grid-viewer {{
                min-height: 250px;
            }}

            .controls-wrapper {{
                gap: 8px;
            }}

            .control-button {{
                padding: 8px;
                min-width: 36px;
            }}

            .control-icon {{
                width: 18px;
                height: 18px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
            <p class="header-info">Viewing {num_animations} animation{("s" if num_animations > 1 else "")} • Synchronized playback</p>
        </div>
        
        <div class="grid-container">
            {viewers_html}
        </div>

        <!-- Global Controls -->
        <div class="controls-container">
            <div class="controls-wrapper">
                <!-- Play/Pause Button -->
                <button id="playPauseBtn" class="control-button" title="Play/Pause All">
                    <svg id="playIcon" class="control-icon" fill="currentColor" viewBox="0 0 24 24" style="display: none;">
                        <path d="M8 5v14l11-7z"/>
                    </svg>
                    <svg id="pauseIcon" class="control-icon" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/>
                    </svg>
                </button>

                <!-- Stop Button -->
                <button id="stopBtn" class="control-button" title="Stop All">
                    <svg class="control-icon" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M6 6h12v12H6z"/>
                    </svg>
                </button>

                <!-- Progress Slider -->
                <input
                    type="range"
                    min="0"
                    max="100"
                    value="0"
                    id="progressSlider"
                    class="progress-slider"
                    title="Seek"
                />

                <!-- Time Display -->
                <div id="timeDisplay" class="time-display">0:00 / 0:00</div>
            </div>
        </div>

        <div class="info-panel">
            <p class="info-text">
                🎬 All animations are synchronized • 🖱️ Use mouse/touch to rotate individual views • 🎮 Global controls at bottom
            </p>
        </div>
    </div>

    <script>
        // Get references to all viewers
        const viewers = [];
        for (let i = 0; i < {num_animations}; i++) {{
            const viewer = document.getElementById(`viewer${{i}}`);
            if (viewer) viewers.push(viewer);
        }}

        // Get references to controls
        const playPauseBtn = document.getElementById('playPauseBtn');
        const stopBtn = document.getElementById('stopBtn');
        const progressSlider = document.getElementById('progressSlider');
        const timeDisplay = document.getElementById('timeDisplay');
        const playIcon = document.getElementById('playIcon');
        const pauseIcon = document.getElementById('pauseIcon');

        // State variables
        let isPlaying = true;
        let isUserDragging = false;
        let maxAnimationDuration = 0;
        let loadedCount = 0;

        // Wait for all viewers to load
        viewers.forEach((viewer, idx) => {{
            viewer.addEventListener('load', () => {{
                loadedCount++;
                
                // Get animation duration
                const duration = viewer.animationDuration || 0;
                maxAnimationDuration = Math.max(maxAnimationDuration, duration);
                
                // When all loaded, log info
                if (loadedCount === viewers.length) {{
                    console.log(`✅ All ${{viewers.length}} animations loaded`);
                    console.log(`Max animation duration: ${{maxAnimationDuration.toFixed(2)}}s`);
                }}
            }});

            viewer.addEventListener('error', (event) => {{
                console.error(`Viewer ${{idx}} loading error:`, event.detail);
            }});
        }});

        // Play/Pause functionality
        playPauseBtn.addEventListener('click', (e) => {{
            e.stopPropagation();
            togglePlayPause();
        }});

        // Stop functionality
        stopBtn.addEventListener('click', (e) => {{
            e.stopPropagation();
            stopPlayback();
        }});

        // Progress slider functionality
        progressSlider.addEventListener('input', (e) => {{
            if (isUserDragging) {{
                const progress = parseFloat(e.target.value) / 100;
                const targetTime = progress * maxAnimationDuration;
                
                // Sync all viewers
                viewers.forEach(viewer => {{
                    viewer.currentTime = targetTime;
                }});
            }}
        }});

        progressSlider.addEventListener('mousedown', () => {{
            isUserDragging = true;
        }});

        progressSlider.addEventListener('mouseup', () => {{
            isUserDragging = false;
        }});

        progressSlider.addEventListener('touchstart', () => {{
            isUserDragging = true;
        }});

        progressSlider.addEventListener('touchend', () => {{
            isUserDragging = false;
        }});

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {{
            if (e.code === 'Space') {{
                e.preventDefault();
                togglePlayPause();
            }} else if (e.code === 'KeyS') {{
                e.preventDefault();
                stopPlayback();
            }}
        }});

        // Update progress slider and time display continuously
        setInterval(() => {{
            if (!isUserDragging && maxAnimationDuration > 0) {{
                // Use the first viewer as reference (they should be synced)
                const currentTime = viewers[0]?.currentTime || 0;
                const progress = (currentTime / maxAnimationDuration) * 100;
                progressSlider.value = progress.toFixed(2);
                
                // Update time display
                const currentMin = Math.floor(currentTime / 60);
                const currentSec = Math.floor(currentTime % 60);
                const durationMin = Math.floor(maxAnimationDuration / 60);
                const durationSec = Math.floor(maxAnimationDuration % 60);
                
                timeDisplay.textContent = 
                    `${{currentMin}}:${{currentSec.toString().padStart(2, '0')}} / ` +
                    `${{durationMin}}:${{durationSec.toString().padStart(2, '0')}}`;
            }}
        }}, 100);

        // Functions
        function togglePlayPause() {{
            if (isPlaying) {{
                viewers.forEach(viewer => viewer.pause());
                isPlaying = false;
                playIcon.style.display = 'block';
                pauseIcon.style.display = 'none';
            }} else {{
                viewers.forEach(viewer => viewer.play());
                isPlaying = true;
                playIcon.style.display = 'none';
                pauseIcon.style.display = 'block';
            }}
        }}

        function stopPlayback() {{
            viewers.forEach(viewer => {{
                viewer.pause();
                viewer.currentTime = 0;
            }});
            progressSlider.value = 0;
            isPlaying = false;
            playIcon.style.display = 'block';
            pauseIcon.style.display = 'none';
        }}

        // Auto-hide controls hint after first interaction
        let hasInteracted = false;
        document.addEventListener('click', () => {{
            if (!hasInteracted) {{
                hasInteracted = true;
            }}
        }}, {{ once: true }});
    </script>
</body>
</html>"""


def create_separate_viewers_grid(
    glb_paths: list[str | Path],
    output_path: str | Path,
    title: str = "Multi-Animation Grid Viewer",
    labels: list[str] | None = None,
    grid_cols: int | None = None,
) -> None:
    """Convenience function to create a grid of separate viewer instances.

    Args:
        glb_paths (List[Union[str, Path]]): List of paths to GLB files
        output_path (Union[str, Path]): Path where HTML will be saved
        title (str): Title for the HTML page
        labels (Optional[List[str]]): Optional labels for each animation
        grid_cols (Optional[int]): Number of columns in grid (auto-calculated if None)
    """
    viewer = SeparateViewersGrid()
    viewer.create_grid_viewer(
        glb_paths=[str(p) for p in glb_paths],
        output_path=str(output_path),
        title=title,
        labels=labels,
        grid_cols=grid_cols,
    )
