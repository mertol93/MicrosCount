# Third-party notices

MicrosCount is licensed under the GNU General Public License v3.0 or later (see `LICENSE`).
It incorporates or bundles the following third-party work.

## SEM_Porosity (MATLAB) by Arash Rabbani — BSD 3-Clause

`src/microscount/materials/porosity.py` is a Python port of `SEM_Porosity.m`, and `tests/data/SEM*.{jpg,png}`
are its sample images and reference outputs.

```
Copyright (c) 2020, Arash Rabbani
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution

* Neither the name of  nor the names of its
  contributors may be used to endorse or promote products derived from this
  software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

The author asks users of SEM_Porosity to cite Rabbani & Salehi (2017) and Ezeakacha et al. (2018);
see `CITATION.cff`.

## ImageJ

The ImageJ "Default" threshold (`AutoThresholder.defaultIsoData` / `IJIsoData`) is re-implemented
from the ImageJ 1.x source code, which is in the public domain. ImageJ itself is not distributed
with MicrosCount; it is used only by the validation harness in `validation/imagej`.

## Libraries bundled in the installers

| Component | Licence |
|---|---|
| Python | Python Software Foundation License |
| Qt 6 and Qt for Python (PySide6) | GNU LGPL v3.0 — the installers ship Qt as unmodified shared libraries, which you may replace; source code is available from https://download.qt.io and https://code.qt.io |
| NumPy | BSD 3-Clause |
| SciPy | BSD 3-Clause |
| scikit-image | BSD 3-Clause |
| tifffile | BSD 3-Clause |
| Pillow | MIT-CMU (HPND) |
| Matplotlib | Matplotlib License (PSF-based, BSD-compatible) |
| PyYAML | MIT |
| openpyxl | MIT |
| pypng | MIT |
| PyInstaller bootloader | GPL v2 with a special exception permitting distribution of bundled applications under any licence |

The full licence texts of these packages are included in their distributions (for example in the
`*.dist-info` folders inside the installed application).
