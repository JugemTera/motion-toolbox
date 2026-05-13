"""HTML viewer for GLB motion animations.

This module provides functionality to create standalone HTML files that embed
GLB animations as base64 data, allowing for self-contained motion visualization.
"""

import base64
from pathlib import Path


class MotionVisualizer:
    """Creates standalone HTML viewers for GLB motion animations.

    This class provides interactive 3D motion visualization by embedding GLB animations
    in HTML files with playback controls.
    """

    def __init__(self):
        """Initialize the HTML viewer generator."""
        pass

    def create_standalone_html(
        self,
        glb_path: str,
        output_path: str,
        title: str = "Motion Animation",
        width: str = "100%",
        height: str = "600px",
    ) -> None:
        """Create a standalone HTML file with embedded GLB animation.

        Args:
            glb_path (str): Path to the GLB file to embed
            output_path (str): Path where the HTML file will be saved
            title (str): Title for the HTML page
            width (str): Width of the viewer (CSS format)
            height (str): Height of the viewer (CSS format)
        """
        # Read and encode GLB file as base64
        glb_base64 = self._encode_glb_to_base64(glb_path)

        # Generate the complete HTML
        html_content = self._generate_html_template(glb_base64=glb_base64, title=title, width=width, height=height)

        # Write HTML file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"✅ Standalone HTML viewer created: {output_path}")

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

    def _generate_html_template(self, glb_base64: str, title: str, width: str, height: str) -> str:
        """Generate Roblox-styled HTML template matching skeleton visualizer.

        Args:
            glb_base64 (str): Base64 encoded GLB data URL
            title (str): Page title
            width (str): Viewer width
            height (str): Viewer height

        Returns:
            str: Complete HTML content
        """
        return f'''<!DOCTYPE html>
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
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #ffffff;
            overflow: hidden;
        }}
        
        .viewer-container {{
            width: 100vw;
            height: 100vh;
            position: relative;
        }}
        
        model-viewer {{
            width: 100%;
            height: 100%;
            background: #ffffff;
            --poster-color: transparent;
        }}
        
        #controls {{
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 12px;
            align-items: center;
            z-index: 100;
            background: rgba(40, 40, 40, 0.85);
            padding: 12px 20px;
            border-radius: 30px;
            backdrop-filter: blur(10px);
        }}
        
        .ctrl-btn {{
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.15);
            border: none;
            cursor: pointer;
            font-size: 14px;
            color: #fff;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.2s;
        }}
        
        .ctrl-btn:hover {{ background: rgba(255, 255, 255, 0.25); }}
        
        .ctrl-btn svg {{
            width: 16px;
            height: 16px;
            fill: currentColor;
        }}
        
        #timeline {{
            width: 200px;
            height: 4px;
            -webkit-appearance: none;
            appearance: none;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 2px;
            outline: none;
        }}
        
        #timeline::-webkit-slider-thumb {{
            -webkit-appearance: none;
            appearance: none;
            width: 14px;
            height: 14px;
            background: #fff;
            border-radius: 50%;
            cursor: pointer;
        }}
        
        #timeline::-moz-range-thumb {{
            width: 14px;
            height: 14px;
            background: #fff;
            border-radius: 50%;
            cursor: pointer;
            border: none;
        }}
        
        #frameInfo {{
            color: #fff;
            font-size: 12px;
            font-weight: 500;
            min-width: 60px;
            text-align: center;
            opacity: 0.9;
        }}
    </style>
</head>
<body>
    <div class="viewer-container">
        <model-viewer
            id="viewer"
            src="{glb_base64}"
            camera-controls
            camera-orbit="0deg 75deg 3m"
            camera-target="0m 0.8m 0m"
            min-camera-orbit="auto auto 1.5m"
            max-camera-orbit="auto auto 10m"
            shadow-intensity="0.3"
            interaction-prompt="none"
            autoplay
        ></model-viewer>
        
        <div id="controls">
            <button class="ctrl-btn" id="playBtn">
                <svg viewBox="0 0 24 24" id="pauseIcon"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>
                <svg viewBox="0 0 24 24" id="playIcon" style="display:none"><path d="M8 5v14l11-7z"/></svg>
            </button>
            <button class="ctrl-btn" id="resetBtn">
                <svg viewBox="0 0 24 24"><path d="M6 6h12v12H6z"/></svg>
            </button>
            <input type="range" id="timeline" min="0" max="100" value="0">
            <div id="frameInfo">0%</div>
        </div>
    </div>
    
    <script type="module">
        const viewer = document.getElementById('viewer');
        const playBtn = document.getElementById('playBtn');
        const resetBtn = document.getElementById('resetBtn');
        const timeline = document.getElementById('timeline');
        const frameInfo = document.getElementById('frameInfo');
        const playIcon = document.getElementById('playIcon');
        const pauseIcon = document.getElementById('pauseIcon');
        
        let isPlaying = true;
        
        viewer.addEventListener('load', () => {{
            // Animation is auto-playing
            updateProgress();
        }});
        
        function updateProgress() {{
            if (viewer.availableAnimations && viewer.availableAnimations.length > 0) {{
                const progress = (viewer.animationTime / viewer.animationDuration) * 100 || 0;
                timeline.value = progress;
                frameInfo.textContent = Math.round(progress) + '%';
            }}
            requestAnimationFrame(updateProgress);
        }}
        
        playBtn.addEventListener('click', () => {{
            isPlaying = !isPlaying;
            viewer.paused = !isPlaying;
            playIcon.style.display = isPlaying ? 'none' : 'block';
            pauseIcon.style.display = isPlaying ? 'block' : 'none';
        }});
        
        resetBtn.addEventListener('click', () => {{
            viewer.currentTime = 0;
            timeline.value = 0;
            frameInfo.textContent = '0%';
        }});
        
        timeline.addEventListener('input', (e) => {{
            const progress = e.target.value / 100;
            viewer.currentTime = progress * viewer.duration;
            frameInfo.textContent = Math.round(e.target.value) + '%';
        }});
    </script>
</body>
</html>'''


def create_motion_visualization(
    glb_path: str, output_path: str, title: str = "Motion Animation", autoplay: bool = False
) -> None:
    """Convenience function to create an HTML viewer for a GLB motion file.

    Args:
        glb_path (str): Path to the GLB file
        output_path (str): Path where HTML will be saved
        title (str): Title for the HTML page
        autoplay (bool): Whether to auto-play the animation
    """
    visualizer = MotionVisualizer()
    visualizer.create_standalone_html(glb_path, output_path, title)


# Keep backward compatibility
GLBHTMLViewer = MotionVisualizer
create_html_viewer = create_motion_visualization
