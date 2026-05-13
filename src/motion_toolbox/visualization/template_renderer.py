"""Simple template rendering utilities for HTML visualization templates."""

from pathlib import Path
from typing import Any


class TemplateRenderer:
    """Renders HTML templates with variable substitution."""

    def __init__(self, template_dir: Path | None = None):
        """Initialize the template renderer.

        Args:
            template_dir: Directory containing templates. Defaults to templates/ in same directory.
        """
        if template_dir is None:
            template_dir = Path(__file__).parent / "templates"
        self.template_dir = Path(template_dir)

    def render(self, template_name: str, context: dict[str, Any]) -> str:
        """Render a template with the given context.

        Args:
            template_name: Name of the template file
            context: Dictionary of template variables

        Returns:
            Rendered HTML string
        """
        template_path = self.template_dir / template_name

        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        with open(template_path) as f:
            template_content = f.read()

        # Replace template variables
        for key, value in context.items():
            placeholder = f"{{{{{key}}}}}"
            # Convert value to string if not already
            if isinstance(value, str):
                template_content = template_content.replace(placeholder, value)
            else:
                template_content = template_content.replace(placeholder, str(value))

        return template_content


def render_ground_plane_code(checkerboard: bool) -> str:
    """Generate JavaScript code for ground plane based on configuration.

    Args:
        checkerboard: Whether to use checkerboard pattern (True) or solid color (False)

    Returns:
        JavaScript code string for ground plane material and geometry
    """
    if checkerboard:
        return """const groundGeometry = new THREE.PlaneGeometry(50, 50);
        
        // Create checkerboard texture
        const checkerboardSize = 50;  // Number of squares (50x50 grid for 1m squares)
        const canvas = document.createElement('canvas');
        canvas.width = checkerboardSize * 2;
        canvas.height = checkerboardSize * 2;
        const ctx = canvas.getContext('2d');
        
        // Draw checkerboard
        const lightColor = '#f5f5f5';  // Light grey
        const darkColor = '#d0d0d0';   // Darker grey
        
        for (let i = 0; i < checkerboardSize; i++) {
            for (let j = 0; j < checkerboardSize; j++) {
                ctx.fillStyle = (i + j) % 2 === 0 ? lightColor : darkColor;
                ctx.fillRect(i * 2, j * 2, 2, 2);
            }
        }
        
        const checkerTexture = new THREE.CanvasTexture(canvas);
        checkerTexture.wrapS = THREE.RepeatWrapping;
        checkerTexture.wrapT = THREE.RepeatWrapping;
        checkerTexture.magFilter = THREE.NearestFilter;  // Crisp edges
        checkerTexture.minFilter = THREE.NearestFilter;
        
        const groundMaterial = new THREE.MeshPhongMaterial({
            map: checkerTexture,
            shininess: 10,
        });"""
    else:
        return """const groundGeometry = new THREE.PlaneGeometry(50, 50);
        const groundMaterial = new THREE.MeshStandardMaterial({
            color: 0xf0f0f0,
            roughness: 0.9,
            metalness: 0.0,
        });"""
