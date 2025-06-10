import warnings

from tueplots import bundles, figsizes, fonts, fontsizes


def jmlr(*, rel_width=1.0, rel_height=None, nrows=1, ncols=1, family="serif", **kwargs):
    if rel_height is not None:
        warnings.warn("The `rel_height` argument is not supported for the JMLR bundle")

    size = figsizes.jmlr2001(
        rel_width=rel_width,
        nrows=nrows,
        ncols=ncols,
        **kwargs,
    )
    font_config = fonts.jmlr2001_tex(family=family)
    fontsize_config = fontsizes.jmlr2001()

    tueplots_rcparams = {**font_config, **size, **fontsize_config}

    _add_latex_preamble(tueplots_rcparams)

    return tueplots_rcparams


def jmlr_docker(*, rel_width=1.0, rel_height=None, nrows=1, ncols=1, family="serif", **kwargs):
    """Docker-compatible version of jmlr bundle that doesn't use LaTeX."""
    if rel_height is not None:
        warnings.warn("The `rel_height` argument is not supported for the JMLR bundle")

    size = figsizes.jmlr2001(
        rel_width=rel_width,
        nrows=nrows,
        ncols=ncols,
        **kwargs,
    )
    # Use a similar non-TeX font configuration (neurips2021 has similar styling)
    font_config = fonts.neurips2021(family=family)
    fontsize_config = fontsizes.jmlr2001()

    tueplots_rcparams = {**font_config, **size, **fontsize_config}

    # Explicitly disable LaTeX
    tueplots_rcparams["text.usetex"] = False

    return tueplots_rcparams


def beamer_moml(*, rel_width=1.0, rel_height=0.8, nrows=1):
    tueplots_rcparams = bundles.beamer_moml(
        rel_width=rel_width,
        rel_height=rel_height,
    )

    tueplots_rcparams["text.usetex"] = True

    _add_latex_preamble(tueplots_rcparams)

    return tueplots_rcparams


def _add_latex_preamble(rcparams):
    if "text.latex.preamble" not in rcparams:
        rcparams["text.latex.preamble"] = ""

    rcparams["text.latex.preamble"] += r"\usepackage{amsfonts}" + "\n"
    rcparams["text.latex.preamble"] += r"\usepackage{siunitx}" + "\n"
    rcparams["text.latex.preamble"] += r"\usepackage{bm}" + "\n"
