"""How to cite MicrosCount and the methods it implements."""

from ._version import __version__

PREFERRED = (
    "Acarer-Arat, S., Pir, İ., Tüfekci, M., Güneş-Durak, S., Akman, A., & Tüfekci, N. (2024). "
    "Heavy Metal Rejection Performance and Mechanical Performance of Cellulose-Nanofibril-Reinforced "
    "Cellulose Acetate Membranes. ACS Omega, 9(41), 42159–42171. https://doi.org/10.1021/acsomega.4c03038"
)

SOFTWARE = (
    f"Tüfekci, M. MicrosCount: open-source microscopy image analysis (version {__version__}). "
    "https://github.com/mertol93/microscount"
)

METHOD_TRANSLOCATION = (
    "Noursadeghi, M., Tsang, J., Haustein, T., Miller, R. F., Chain, B. M., & Katz, D. R. (2008). "
    "Quantitative imaging assay for NF-κB nuclear translocation in primary human macrophages. "
    "Journal of Immunological Methods, 329(1–2), 194–200. https://doi.org/10.1016/j.jim.2007.10.015"
)

METHOD_POROSITY = (
    "Rabbani, A., & Salehi, S. (2017). Dynamic modeling of the formation damage and mud cake deposition "
    "using filtration theories coupled with SEM image processing. Journal of Natural Gas Science and "
    "Engineering, 42, 157–168. https://doi.org/10.1016/j.jngse.2017.02.047\n"
    "Ezeakacha, C. P., Rabbani, A., Salehi, S., & Ghalambor, A. (2018, February). Integrated Image Processing "
    "and Computational Techniques to Characterize Formation Damage. In SPE International Conference and "
    "Exhibition on Formation Damage Control. Society of Petroleum Engineers. "
    "https://onepetro.org/SPEFD/proceedings-abstract/18FD/1-18FD/D012S007R004/214874"
)

BIBTEX = r"""@article{AcarerArat2024,
  author  = {Acarer-Arat, Seren and Pir, {\.I}nci and T{\"u}fekci, Mertol and G{\"u}ne{\c{s}}-Durak, Sevgi and Akman, Alp and T{\"u}fekci, Ne{\c{s}}e},
  title   = {Heavy Metal Rejection Performance and Mechanical Performance of Cellulose-Nanofibril-Reinforced Cellulose Acetate Membranes},
  journal = {ACS Omega},
  year    = {2024},
  volume  = {9},
  number  = {41},
  pages   = {42159--42171},
  doi     = {10.1021/acsomega.4c03038}
}"""


def citation_text(module: str | None = None) -> str:
    """Plain-text citation request written next to every set of results."""
    lines = [
        "If you use MicrosCount in published work, please cite:",
        "",
        "  " + PREFERRED,
        "",
        "and the software itself:",
        "",
        "  " + SOFTWARE,
        "",
    ]
    if module in (None, "translocation"):
        lines += ["Nuclear translocation (paper method) is based on:", "", "  " + METHOD_TRANSLOCATION, ""]
    if module in (None, "porosity"):
        lines += [
            "SEM porosity is a port of A. Rabbani's SEM_Porosity MATLAB code (BSD-3-Clause); its author asks you to cite:",
            "",
        ]
        lines += ["  " + ln for ln in METHOD_POROSITY.split("\n")]
        lines += [""]
    lines += ["BibTeX for the preferred citation:", "", BIBTEX, ""]
    return "\n".join(lines)
