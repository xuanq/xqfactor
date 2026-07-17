import subprocess
import sys


def test_analysis_base_import_does_not_load_optional_statistics_packages() -> None:
    """导入 analysis 基础接口时不应加载 SciPy 或 statsmodels。"""
    command = """
import sys
import xqfactor.analysis

assert 'scipy' not in sys.modules
assert 'statsmodels' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", command], check=True)
