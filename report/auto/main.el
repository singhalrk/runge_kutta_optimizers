(TeX-add-style-hook
 "main"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "12pt" "twoside")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("natbib" "square" "sort" "comma" "numbers")))
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperref")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperimage")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "hyperbaseurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "nolinkurl")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "url")
   (add-to-list 'LaTeX-verbatim-macros-with-braces-local "path")
   (add-to-list 'LaTeX-verbatim-macros-with-delims-local "path")
   (TeX-run-style-hooks
    "latex2e"
    "introduction"
    "grad_flow"
    "integration"
    "rk2Ralston"
    "rk2_heun"
    "deep_learning"
    "biblio"
    "experiments"
    "article"
    "art12"
    "graphicx"
    "amsmath"
    "amssymb"
    "natbib"
    "verbatim"
    "floatpag"
    "subeqnarray"
    "mathrsfs"
    "cancel"
    "subcaption"
    "hyperref"
    "wrapfig"
    "amsthm"
    "bbm"
    "times")
   (TeX-add-symbols
    '("norm" 1)
    "signaturerule")
   (LaTeX-add-amsthm-newtheorems
    "theorem"
    "lemma"
    "prop"))
 :latex)

