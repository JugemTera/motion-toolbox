"""Simple Standalone Test: Unified Visualizer

This script runs the unified visualizer unit tests and writes an HTML
report summarising the results.

USAGE:
    # From the repository root
    PYTHONPATH=src python3 examples/test_visualizer_simple.py
"""

import subprocess
from datetime import datetime
from pathlib import Path


def run_tests_and_generate_report():
    """Run unit tests and generate HTML report."""
    print("=" * 80)
    print("  UNIFIED VISUALIZER - TEST REPORT GENERATOR")
    print("=" * 80)
    print()
    
    output_dir = Path("outputs/test_report")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Running unit tests...")
    print()
    
    # Run pytest
    result = subprocess.run(
        ["python3", "-m", "pytest", "tests/test_unified_visualizer.py", "-v", "--tb=short"],
        capture_output=True,
        text=True
    )
    
    # Parse output
    test_output = result.stdout + result.stderr
    tests_passed = result.returncode == 0
    
    # Count tests
    import re
    passed_match = re.search(r'(\d+) passed', test_output)
    failed_match = re.search(r'(\d+) failed', test_output)
    
    num_passed = int(passed_match.group(1)) if passed_match else 0
    num_failed = int(failed_match.group(1)) if failed_match else 0
    total_tests = num_passed + num_failed
    
    print(f"Tests completed: {num_passed} passed, {num_failed} failed")
    print()
    
    # Generate HTML report
    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Unified Visualizer - Test Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px 40px;
            background: #f8f9fa;
        }}
        .stat {{
            text-align: center;
            padding: 20px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: {'#28a745' if tests_passed else '#dc3545'};
        }}
        .stat-label {{
            font-size: 0.9em;
            color: #666;
            margin-top: 5px;
        }}
        .content {{
            padding: 40px;
        }}
        .test-output {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 12px;
            overflow-x: auto;
        }}
        pre {{
            white-space: pre-wrap;
            word-wrap: break-word;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }}
        .capabilities {{
            background: #f8f9fa;
            padding: 30px;
            border-radius: 12px;
            margin-top: 30px;
        }}
        .capabilities h2 {{
            color: #667eea;
            margin-bottom: 20px;
        }}
        .capabilities ul {{
            list-style: none;
            columns: 2;
            column-gap: 40px;
        }}
        .capabilities li {{
            padding: 10px 0;
            border-bottom: 1px solid #e0e0e0;
        }}
        .capabilities li:before {{
            content: "✓";
            color: #28a745;
            font-weight: bold;
            margin-right: 10px;
        }}
        .footer {{
            background: #f8f9fa;
            padding: 30px 40px;
            text-align: center;
            border-top: 1px solid #e0e0e0;
            color: #666;
        }}
        .status-badge {{
            display: inline-block;
            padding: 10px 20px;
            border-radius: 6px;
            font-weight: bold;
            margin: 20px 0;
        }}
        .status-success {{
            background: #d4edda;
            color: #155724;
        }}
        .status-failure {{
            background: #f8d7da;
            color: #721c24;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎨 Unified Visualizer Test Report</h1>
            <p>Comprehensive test suite validation</p>
            <div class="status-badge status-{'success' if tests_passed else 'failure'}">
                {'✓ ALL TESTS PASSED' if tests_passed else '✗ SOME TESTS FAILED'}
            </div>
        </div>
        
        <div class="stats">
            <div class="stat">
                <div class="stat-value">{total_tests}</div>
                <div class="stat-label">Total Tests</div>
            </div>
            <div class="stat">
                <div class="stat-value">{num_passed}</div>
                <div class="stat-label">Tests Passed</div>
            </div>
            <div class="stat">
                <div class="stat-value">{num_failed}</div>
                <div class="stat-label">Tests Failed</div>
            </div>
            <div class="stat">
                <div class="stat-value">{int((num_passed/total_tests*100) if total_tests > 0 else 0)}%</div>
                <div class="stat-label">Success Rate</div>
            </div>
        </div>
        
        <div class="content">
            <h2 style="color: #667eea; margin-bottom: 20px;">Test Output</h2>
            <div class="test-output">
                <pre>{test_output}</pre>
            </div>
            
            <div class="capabilities">
                <h2>✨ Features Tested</h2>
                <ul>
                    <li>Format detection (263D, 272D, 205D, 232D ALIGN, Motion, joints)</li>
                    <li>Compatibility matrix validation</li>
                    <li>SMPL mesh visualization support</li>
                    <li>MHR mesh visualization</li>
                    <li>Skeleton visualization (all formats)</li>
                    <li>AUTO mode intelligence</li>
                    <li>Fallback behavior (on_incompatible)</li>
                    <li>Scene data export (JSON for frontends)</li>
                </ul>
            </div>
        </div>
        
        <div class="footer">
            <p><strong>Unified Visualizer System</strong></p>
            <p style="margin-top: 10px;">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p style="margin-top: 10px; font-size: 0.9em;">
                Full test suite with {total_tests} test cases
            </p>
        </div>
    </div>
</body>
</html>
"""
    
    report_path = output_dir / "test_report.html"
    report_path.write_text(report_html)
    
    print("=" * 80)
    print("  REPORT GENERATED")
    print("=" * 80)
    print()
    print(f"✓ HTML report: {report_path.absolute()}")
    print()
    print("Open in browser:")
    print(f"  file://{report_path.absolute()}")
    print()
    print("Or run:")
    print(f"  open {report_path}  # macOS")
    print(f"  xdg-open {report_path}  # Linux")
    print()
    
    return report_path

if __name__ == '__main__':
    run_tests_and_generate_report()
