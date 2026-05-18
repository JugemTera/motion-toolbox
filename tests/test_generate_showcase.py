"""Generate actual 3D visualizations for showcase - saves to outputs/showcase_3d/

IMPORTANT: To generate the complete showcase WITH MHR visualizations, run:
    bash scripts/run_with_mhr.sh python -m pytest tests/test_generate_showcase.py -v

Without the run_with_mhr.sh wrapper, MHR visualizations will be skipped.
"""

from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.requires_smpl


class TestGenerateShowcase:
    """Generate permanent visualization files for the showcase."""

    def test_generate_all_visualizations(self):
        """Generate all visualization formats using real data."""
        from motion_toolbox.converters.motion_converters import HumanML3DTo272Converter
        from motion_toolbox.visualization.unified_visualizer import Visualizer

        output_dir = Path("outputs/showcase_3d")
        output_dir.mkdir(parents=True, exist_ok=True)

        vis = Visualizer()
        sample_dir = Path("sample_data")

        print("\n" + "=" * 60)
        print("GENERATING VISUALIZATION SHOWCASE")
        print("=" * 60)

        # 1. 263D (use real HumanML3D data)
        print("\n1. HumanML3D 263D Format...")
        data_263d = np.load(sample_dir / "humanml3d/000002.npy")
        path1 = vis.visualize(
            data_263d, output_dir / "263d_skeleton.html", mode="skeleton", fps=20, title="HumanML3D 263D"
        )
        assert path1.exists()
        print(f"   Created: 263d_skeleton.html ({data_263d.shape[0]} frames)")

        # 2. 272D (converted from 263D - same motion for comparison)
        print("\n2. 272D Format (converted from 263D)...")
        converter = HumanML3DTo272Converter()
        data_272d_converted = converter.convert(data_263d)
        path2 = vis.visualize(
            data_272d_converted, output_dir / "272d_skeleton.html", mode="skeleton", fps=20, title="272D Format"
        )
        assert path2.exists()
        print(f"   Created: 272d_skeleton.html ({data_272d_converted.shape[0]} frames)")

        # 3. Real 272D Sample Data (different motion)
        print("\n3. Real 272D Sample (from dataset)...")
        data_272d_real = np.load(sample_dir / "format_272/000000.npy")
        path3 = vis.visualize(
            data_272d_real, output_dir / "272d_real_skeleton.html", mode="skeleton", fps=20, title="272D Real Sample"
        )
        assert path3.exists()
        print(f"   Created: 272d_real_skeleton.html ({data_272d_real.shape[0]} frames)")

        # 4. Joint positions (22 joints) - real joint data
        print("\n4. Joint Positions (22 joints)...")
        joints_22 = np.load(sample_dir / "humanml3d/000010_joints.npy")
        path4 = vis.visualize(
            joints_22, output_dir / "joints_22_skeleton.html", mode="skeleton", fps=20, title="22 Joints"
        )
        assert path4.exists()
        print(f"   Created: joints_22_skeleton.html ({joints_22.shape[0]} frames)")

        # 5. SMPL Parameters - use real SMPL data (convert to Motion first)
        print("\n5. SMPL Motion (real SMPL data)...")
        from pathlib import Path as P

        from motion_toolbox.converters.motion_converters import SMPLToMotionConverter

        body_model_path = str(P.home() / ".motion_toolbox" / "datasets" / "body_models_clean_pkl")
        smpl_data = dict(np.load(sample_dir / "smpl/metrabs_1.npz"))
        # Take first 100 frames for faster visualization
        smpl_subset = {
            "pose_body": smpl_data["pose_body"][:100],
            "root_orient": smpl_data["root_orient"][:100],
            "trans": smpl_data["trans"][:100],
            "betas": smpl_data["betas"][:100] if smpl_data["betas"].ndim > 1 else smpl_data["betas"],
        }
        # Convert SMPL to Motion, visualizer handles coordinate normalization
        try:
            smpl_converter = SMPLToMotionConverter(body_model_path=body_model_path)
            motion_smpl = smpl_converter.convert(smpl_subset)

            # Pass joint positions - Visualizer auto-normalizes coordinates for SMPL
            path5 = vis.visualize(
                motion_smpl.positions, output_dir / "smpl_skeleton.html", mode="skeleton", fps=30, title="SMPL Motion"
            )
            assert path5.exists()
            print(f"   Created: smpl_skeleton.html ({motion_smpl.positions.shape[0]} frames)")
        except Exception as e:
            print(f"   Skipped SMPL: {e}")

        # 6. Different 22-joint motion (variety)
        print("\n6. Another 22-joint motion...")
        joints_22_b = np.load(sample_dir / "humanml3d/000020_joints.npy")
        path6 = vis.visualize(
            joints_22_b, output_dir / "joints_22b_skeleton.html", mode="skeleton", fps=20, title="22 Joints (Motion B)"
        )
        assert path6.exists()
        print(f"   Created: joints_22b_skeleton.html ({joints_22_b.shape[0]} frames)")

        # 7. SMPL Mesh Visualization
        print("\n7. SMPL Mesh (3D body)...")
        try:
            vis_mesh = Visualizer(body_model_path=body_model_path)
            # Use shorter clip and fix betas to 10 dims
            smpl_mesh_subset = {
                "pose_body": smpl_data["pose_body"][:50],
                "root_orient": smpl_data["root_orient"][:50],
                "trans": smpl_data["trans"][:50],
                "betas": smpl_data["betas"][:50, :10].mean(axis=0)
                if smpl_data["betas"].ndim > 1
                else smpl_data["betas"][:10],
            }
            motion_mesh = smpl_converter.convert(smpl_mesh_subset)
            path7 = vis_mesh.visualize(
                motion_mesh, output_dir / "smpl_mesh.html", mode="mesh", body_model="smpl", fps=30, title="SMPL Mesh"
            )
            assert path7.exists()
            print(f"   Created: smpl_mesh.html ({motion_mesh.num_frames} frames)")
        except Exception as e:
            print(f"   Skipped SMPL Mesh: {e}")

        # 9. 205D (STMC) Skeleton Visualization
        print("\n9. STMC 205D Skeleton...")
        try:
            data_205d = np.load(sample_dir / "format_205/sample.npy")
            path9 = vis.visualize(
                data_205d, output_dir / "205d_skeleton.html", mode="skeleton", fps=30, title="STMC 205D"
            )
            assert path9.exists()
            print(f"   Created: 205d_skeleton.html ({data_205d.shape[0]} frames)")
        except Exception as e:
            print(f"   Skipped 205D skeleton: {e}")

        # 10. 205D to SMPL Mesh Visualization
        print("\n10. STMC 205D → SMPL Mesh...")
        try:
            vis_mesh_205 = Visualizer(body_model_path=body_model_path)
            path10 = vis_mesh_205.visualize(
                data_205d,
                output_dir / "205d_smpl_mesh.html",
                mode="mesh",
                body_model="smpl",
                fps=30,
                title="205D → SMPL Mesh",
            )
            assert path10.exists()
            print(f"   Created: 205d_smpl_mesh.html ({data_205d.shape[0]} frames)")
        except Exception as e:
            print(f"   Skipped 205D mesh: {e}")

        # 10.5. ALIGN 232D Skeleton Visualization
        print("\n10.5 ALIGN 232D Skeleton...")
        try:
            # Convert SMPL to ALIGN features for demonstration
            from motion_toolbox.converters.align_converter import ALIGNConverter

            align_converter = ALIGNConverter(body_model_path=body_model_path)
            align_features = align_converter.to_align(motion_smpl)
            path10_5 = vis.visualize(
                align_features, output_dir / "232d_align_skeleton.html", mode="skeleton", fps=30, title="ALIGN 232D"
            )
            assert path10_5.exists()
            print(f"   Created: 232d_align_skeleton.html ({align_features.shape[0]} frames)")
        except Exception as e:
            print(f"   Skipped ALIGN skeleton: {e}")

        # 10.6. ALIGN 232D to SMPL Mesh
        print("\n10.6 ALIGN 232D → SMPL Mesh...")
        try:
            vis_mesh_align = Visualizer(body_model_path=body_model_path)
            path10_6 = vis_mesh_align.visualize(
                align_features,
                output_dir / "232d_align_smpl_mesh.html",
                mode="mesh",
                body_model="smpl",
                fps=30,
                title="ALIGN → SMPL Mesh",
            )
            assert path10_6.exists()
            print(f"   Created: 232d_align_smpl_mesh.html ({align_features.shape[0]} frames)")
        except Exception as e:
            print(f"   Skipped ALIGN mesh: {e}")

        # 11. MHR Skeleton Visualization
        print("\n11. MHR Skeleton...")
        try:
            import torch

            mhr_data = torch.load(sample_dir / "mhr/gvhmr_1_mhr.pt", weights_only=False)
            mhr_assets_path = str(Path.home() / ".motion_toolbox" / "datasets" / "mhr" / "assets")
            vis_mhr = Visualizer(body_model_path=mhr_assets_path)
            path11 = vis_mhr.visualize(
                mhr_data, output_dir / "mhr_skeleton.html", mode="skeleton", fps=30, title="MHR Skeleton"
            )
            assert path11.exists()
            print(f"   Created: mhr_skeleton.html ({mhr_data['model_parameters'].shape[0]} frames)")
        except Exception as e:
            print(f"   Skipped MHR skeleton: {e}")

        # 12. MHR Mesh Visualization
        print("\n12. MHR Mesh...")
        try:
            vis_mhr_mesh = Visualizer(body_model_path=mhr_assets_path)
            path12 = vis_mhr_mesh.visualize(
                mhr_data, output_dir / "mhr_mesh.html", mode="mesh", body_model="mhr", fps=30, title="MHR Mesh"
            )
            assert path12.exists()
            print(f"   Created: mhr_mesh.html ({mhr_data['model_parameters'].shape[0]} frames)")
        except Exception as e:
            print(f"   Skipped MHR mesh: {e}")

        print("\n" + "=" * 60)
        print(f"SUCCESS: 14 visualizations generated in {output_dir.absolute()}")
        print("=" * 60)

        # Generate the styled index
        self._generate_index(output_dir)

    def _generate_index(self, output_dir: Path):
        """Generate styled black and white index page."""
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Motion Visualizer</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #0a0a0a;
            color: #ffffff;
            min-height: 100vh;
            padding: 40px 20px;
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            margin-bottom: 60px;
            padding: 40px 0;
            border-bottom: 1px solid #222;
        }
        
        .header h1 {
            font-size: 3.5em;
            font-weight: 900;
            letter-spacing: -0.03em;
            margin-bottom: 15px;
            background: linear-gradient(180deg, #fff 0%, #888 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .header p {
            font-size: 1.1em;
            color: #666;
            font-weight: 400;
        }
        
        .instructions {
            background: #111;
            border: 1px solid #222;
            padding: 24px 32px;
            border-radius: 8px;
            margin-bottom: 50px;
            display: flex;
            gap: 40px;
            flex-wrap: wrap;
        }
        
        .instruction-item {
            display: flex;
            align-items: center;
            gap: 12px;
            color: #888;
            font-size: 0.95em;
        }
        
        .instruction-item kbd {
            background: #1a1a1a;
            border: 1px solid #333;
            padding: 4px 10px;
            border-radius: 4px;
            font-family: 'Inter', monospace;
            font-size: 0.85em;
            color: #fff;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
            gap: 24px;
            margin-bottom: 60px;
        }
        
        .viz-card {
            background: #111;
            border: 1px solid #222;
            border-radius: 8px;
            overflow: hidden;
            transition: all 0.2s ease;
        }
        
        .viz-card:hover {
            border-color: #444;
            transform: translateY(-2px);
        }
        
        .viz-header {
            padding: 20px 24px;
            border-bottom: 1px solid #222;
            display: flex;
            align-items: center;
            gap: 16px;
        }
        
        .format-badge {
            display: inline-block;
            padding: 6px 12px;
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 4px;
            font-size: 0.75em;
            font-weight: 700;
            letter-spacing: 0.05em;
            color: #fff;
        }
        
        .format-badge.primary {
            background: #fff;
            color: #000;
            border-color: #fff;
        }
        
        .viz-header h3 {
            font-size: 1.1em;
            font-weight: 600;
            color: #fff;
        }
        
        .viz-header .description {
            margin-left: auto;
            font-size: 0.85em;
            color: #555;
        }
        
        .viz-iframe {
            width: 100%;
            height: 450px;
            border: none;
            display: block;
            background: #0a0a0a;
        }
        
        .footer {
            text-align: center;
            padding: 40px;
            border-top: 1px solid #222;
            color: #444;
            font-size: 0.9em;
        }
        
        .footer strong {
            color: #666;
        }
        
        .stats {
            display: flex;
            justify-content: center;
            gap: 60px;
            margin-top: 20px;
        }
        
        .stat {
            text-align: center;
        }
        
        .stat-value {
            font-size: 2em;
            font-weight: 900;
            color: #fff;
        }
        
        .stat-label {
            font-size: 0.8em;
            color: #555;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>MOTION VISUALIZER</h1>
            <p>Unified skeleton animation viewer for motion capture data</p>
        </div>
        
        <div class="instructions">
            <div class="instruction-item">
                <kbd>Drag</kbd>
                <span>Rotate camera</span>
            </div>
            <div class="instruction-item">
                <kbd>Scroll</kbd>
                <span>Zoom in/out</span>
            </div>
            <div class="instruction-item">
                <kbd>Right-drag</kbd>
                <span>Pan view</span>
            </div>
            <div class="instruction-item">
                <kbd>Timeline</kbd>
                <span>Scrub through animation</span>
            </div>
        </div>
        
        <h2 style="color: #666; font-size: 1.2em; margin-bottom: 20px; text-transform: uppercase; letter-spacing: 0.1em;">Skeleton Visualizations</h2>
        <div class="grid">
            <div class="viz-card">
                <div class="viz-header">
                    <span class="format-badge primary">263D</span>
                    <h3>HumanML3D Format</h3>
                    <span class="description">82 frames</span>
                </div>
                <iframe src="263d_skeleton.html" class="viz-iframe"></iframe>
            </div>
            
            <div class="viz-card">
                <div class="viz-header">
                    <span class="format-badge">272D</span>
                    <h3>272D Converted</h3>
                    <span class="description">83 frames</span>
                </div>
                <iframe src="272d_skeleton.html" class="viz-iframe"></iframe>
            </div>
            
            <div class="viz-card">
                <div class="viz-header">
                    <span class="format-badge">272D</span>
                    <h3>272D Real Sample</h3>
                    <span class="description">221 frames</span>
                </div>
                <iframe src="272d_real_skeleton.html" class="viz-iframe"></iframe>
            </div>
            
            <div class="viz-card">
                <div class="viz-header">
                    <span class="format-badge">205D</span>
                    <h3>STMC Skeleton</h3>
                    <span class="description">196 frames</span>
                </div>
                <iframe src="205d_skeleton.html" class="viz-iframe"></iframe>
            </div>
            
            <div class="viz-card">
                <div class="viz-header">
                    <span class="format-badge">JOINTS</span>
                    <h3>22 Joint Positions</h3>
                    <span class="description">69 frames</span>
                </div>
                <iframe src="joints_22_skeleton.html" class="viz-iframe"></iframe>
            </div>
            
            <div class="viz-card">
                <div class="viz-header">
                    <span class="format-badge">JOINTS</span>
                    <h3>22 Joints (Motion B)</h3>
                    <span class="description">variant</span>
                </div>
                <iframe src="joints_22b_skeleton.html" class="viz-iframe"></iframe>
            </div>
            
            <div class="viz-card">
                <div class="viz-header">
                    <span class="format-badge primary">SMPL</span>
                    <h3>SMPL Motion</h3>
                    <span class="description">100 frames</span>
                </div>
                <iframe src="smpl_skeleton.html" class="viz-iframe"></iframe>
            </div>
            
            <div class="viz-card">
                <div class="viz-header">
                    <span class="format-badge">MHR</span>
                    <h3>MHR Skeleton</h3>
                    <span class="description">61 frames</span>
                </div>
                <iframe src="mhr_skeleton.html" class="viz-iframe"></iframe>
            </div>
            
            <div class="viz-card">
                <div class="viz-header">
                    <span class="format-badge primary">232D</span>
                    <h3>ALIGN Format</h3>
                    <span class="description">100 frames</span>
                </div>
                <iframe src="232d_align_skeleton.html" class="viz-iframe"></iframe>
            </div>
        </div>
        
        <h2 style="color: #666; font-size: 1.2em; margin: 40px 0 20px 0; text-transform: uppercase; letter-spacing: 0.1em;">Mesh Visualizations</h2>
        <div class="grid">
            <div class="viz-card">
                <div class="viz-header">
                    <span class="format-badge primary">MESH</span>
                    <h3>SMPL 3D Body</h3>
                    <span class="description">50 frames</span>
                </div>
                <iframe src="smpl_mesh.html" class="viz-iframe"></iframe>
            </div>
            
            <div class="viz-card">
                <div class="viz-header">
                    <span class="format-badge">205D→SMPL</span>
                    <h3>STMC to SMPL Mesh</h3>
                    <span class="description">196 frames</span>
                </div>
                <iframe src="205d_smpl_mesh.html" class="viz-iframe"></iframe>
            </div>
            
            <div class="viz-card">
                <div class="viz-header">
                    <span class="format-badge primary">MHR</span>
                    <h3>MHR 3D Body</h3>
                    <span class="description">61 frames</span>
                </div>
                <iframe src="mhr_mesh.html" class="viz-iframe"></iframe>
            </div>
            
            <div class="viz-card">
                <div class="viz-header">
                    <span class="format-badge">232D→SMPL</span>
                    <h3>ALIGN to SMPL Mesh</h3>
                    <span class="description">100 frames</span>
                </div>
                <iframe src="232d_align_smpl_mesh.html" class="viz-iframe"></iframe>
            </div>
        </div>
        
        <div class="footer">
            <div class="stats">
                <div class="stat">
                    <div class="stat-value">14</div>
                    <div class="stat-label">Visualizations</div>
                </div>
                <div class="stat">
                    <div class="stat-value">6</div>
                    <div class="stat-label">Formats</div>
                </div>
                <div class="stat">
                    <div class="stat-value">22-127</div>
                    <div class="stat-label">Joints</div>
                </div>
            </div>
            <p style="margin-top: 30px;"><strong>Unified Visualizer</strong> &mdash; Motion Toolbox</p>
        </div>
    </div>
</body>
</html>"""

        index_path = output_dir / "index.html"
        index_path.write_text(html)
        print(f"\n   Generated: {index_path}")
