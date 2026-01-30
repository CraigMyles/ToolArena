#! /bin/bash
set -e

git clone https://github.com/KatherLab/STAMP /workspace/STAMP
cd /workspace/STAMP && git checkout 97522aa

apt update && apt install -y libgl1 libglx-mesa0 libglib2.0-0
pip install -e "/workspace/STAMP[ctranspath]"

# Fix: PyTorch 2.6+ changed torch.load() default to weights_only=True
# Add numpy types to safe globals for checkpoint loading
# This must be done BEFORE any checkpoint loading happens

# Create a Python script that patches torch at import time
cat > /workspace/STAMP/torch_compat_patch.py << 'EOF'
"""
Compatibility fix for PyTorch 2.6+ weights_only=True default.
Add numpy types to torch safe globals.
"""
import torch.serialization
import numpy as np

# Add all numpy types that might appear in checkpoints
safe_types = [np.ndarray, np.dtype]

# Add numpy scalar
try:
    from numpy._core.multiarray import scalar as np_scalar
    safe_types.append(np_scalar)
except ImportError:
    try:
        from numpy.core.multiarray import scalar as np_scalar
        safe_types.append(np_scalar)
    except ImportError:
        pass

# Add numpy dtypes
try:
    safe_types.extend([
        np.float16, np.float32, np.float64,
        np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.bool_, np.str_, np.bytes_,
    ])
except Exception:
    pass

# Try adding StrDType for newer numpy
try:
    from numpy.dtypes import StrDType
    safe_types.append(StrDType)
except Exception:
    pass

# Add packaging module types (used in some checkpoints)
try:
    from packaging.version import Version
    from packaging._structures import InfinityType, NegativeInfinityType
    safe_types.extend([Version, InfinityType, NegativeInfinityType])
except Exception:
    pass

# Try all packaging types
try:
    import packaging._structures
    for name in dir(packaging._structures):
        obj = getattr(packaging._structures, name)
        if isinstance(obj, type):
            safe_types.append(obj)
except Exception:
    pass

# Add collections types
try:
    from collections import OrderedDict
    safe_types.append(OrderedDict)
except Exception:
    pass

# Add pathlib types (required for PyTorch Lightning checkpoints)
try:
    from pathlib import PosixPath, PurePosixPath, WindowsPath, PureWindowsPath, Path
    safe_types.extend([PosixPath, PurePosixPath, WindowsPath, PureWindowsPath, Path])
except Exception:
    pass

torch.serialization.add_safe_globals(safe_types)
print("torch_compat_patch: Added safe globals for checkpoint loading")
EOF

# Import the patch in STAMP's main entry points
sed -i '1i import sys; sys.path.insert(0, "/workspace/STAMP"); import torch_compat_patch  # PyTorch 2.6+ fix' /workspace/STAMP/src/stamp/__main__.py
sed -i '1i import sys; sys.path.insert(0, "/workspace/STAMP"); import torch_compat_patch  # PyTorch 2.6+ fix' /workspace/STAMP/src/stamp/modeling/train.py
