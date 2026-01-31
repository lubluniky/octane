# Publishing Octane to PyPI

This guide explains how to publish the Octane Python package to PyPI.

## Prerequisites

1. Install maturin:
```bash
pip install maturin
```

2. Create PyPI account at https://pypi.org/account/register/

3. Create API token at https://pypi.org/manage/account/token/

4. Save token to `~/.pypirc`:
```ini
[pypi]
username = __token__
password = pypi-YOUR_TOKEN_HERE
```

## Local Development

### Build wheel locally
```bash
# Build for current Python version
maturin build --release

# Build with specific features
maturin build --release --features python,metal
```

### Install locally for testing
```bash
# Install in current virtualenv
maturin develop --release

# Test in Python
python -c "import octane; print(octane.__version__)"
```

## Publishing to PyPI

### Option 1: Publish directly
```bash
# Publish to PyPI (requires ~/.pypirc)
maturin publish --features python
```

### Option 2: Build and upload separately
```bash
# Build wheels for all platforms
maturin build --release --features python

# Upload with twine
pip install twine
twine upload target/wheels/*
```

## Building for Multiple Platforms

### Using GitHub Actions (Recommended)

Create `.github/workflows/pypi.yml`:

```yaml
name: PyPI Release

on:
  release:
    types: [published]
  workflow_dispatch:

jobs:
  build:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install Rust
        uses: dtolnay/rust-action@stable

      - name: Install maturin
        run: pip install maturin

      - name: Build wheel
        run: maturin build --release --features python

      - name: Upload wheel
        uses: actions/upload-artifact@v4
        with:
          name: wheel-${{ matrix.os }}-py${{ matrix.python-version }}
          path: target/wheels/*.whl

  publish:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: actions/download-artifact@v4
        with:
          pattern: wheel-*
          merge-multiple: true
          path: wheels

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          packages-dir: wheels/
          password: ${{ secrets.PYPI_API_TOKEN }}
```

### Building for Apple Silicon (M1/M2/M3/M4)

```bash
# Native build on macOS ARM64
maturin build --release --features python,metal,simd

# Cross-compile from x86_64 to ARM64
rustup target add aarch64-apple-darwin
maturin build --release --target aarch64-apple-darwin --features python
```

### Building for Linux

```bash
# Using manylinux for compatibility
maturin build --release --features python --manylinux 2014
```

## Usage in Python

After installation:

```python
import octane

# Create device
device = octane.Device.cpu()  # or .metal() or .cuda(0)

# Trading metrics
metrics = octane.TradingMetrics(window_size=252)
for ret in daily_returns:
    metrics.add_return(ret)

print(f"Sharpe: {metrics.sharpe_ratio():.2f}")
print(f"Max DD: {metrics.max_drawdown():.2%}")
print(f"VaR 95%: {metrics.var(0.95):.2%}")

# Drawdown controller
dd = octane.DrawdownController(
    max_drawdown=0.20,      # Stop at 20% DD
    recovery_threshold=0.10, # Enter recovery at 10% DD
    recovery_risk_factor=0.5 # Halve position size in recovery
)

# Position sizing
sizer = octane.PositionSizer(method="half_kelly", max_position=1.0)
position = sizer.calculate(win_rate=0.6, avg_win=0.02, avg_loss=0.01)
```

## Versioning

Update version in:
1. `Cargo.toml` - `version = "x.y.z"`
2. `pyproject.toml` - `version = "x.y.z"`
3. `src/python/mod.rs` - `m.add("__version__", "x.y.z")?;`
4. `python/octane/__init__.py` - `__version__ = "x.y.z"`

## Troubleshooting

### "maturin: command not found"
```bash
pip install maturin
# or
cargo install maturin
```

### Wheel not installing
Make sure you're building for the correct Python version:
```bash
maturin build --release --interpreter python3.11
```

### Metal/CUDA features not working
These require building on the target platform with appropriate SDKs installed.
