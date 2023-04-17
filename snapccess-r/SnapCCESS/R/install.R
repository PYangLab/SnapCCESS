#' install_SnapCCESS
#'
#' install related python packages,
#' this should be run only once when you install the R pacakge.
#'
#' @param envname The name, or full path, of the environment in which Python packages are to be installed.
#' When NULL (the default), the active environment as set by the RETICULATE_PYTHON_ENV variable will be used;
#' if that is unset, then the r-reticulate environment will be used.
#' @param method Installation method. By default, "auto" automatically finds a method that
#' will work in the local environment. Change the default to force a specific installation method.
#' Note that the "virtualenv" method is not available on Windows.
#' @param conda The path to a conda executable.
#' By default, reticulate will check the PATH,
#' as well as other standard locations for Anaconda installations.
#' @param ... Additional arguments passed to reticulate::py_install function
#'
#' @export
#'
#' @import reticulate rstudioapi
#'
#'
#' @examples #install_SnapCCESS(envname = "SnapCCESS",method = "conda",conda="auto")
#'
install_SnapCCESS <- function(envname = "SnapCCESS",method = "conda",conda="auto",...){
  reticulate::py_install(envname = envname,
             method = method,
             pip=TRUE,
             pip_ignore_installed = TRUE,
             packages = "snapccess",
             python_version = "3.8",
             conda=conda,
             ...)
  rstudioapi::restartSession()
}



# pip_options="--index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "
# reticulate::py_install(
#   packages       = "snapccess",
#   method         = "conda",
#   envname = "snapccess-ylj",
#   python_version = "3.8",
#   conda          = "/home/lijiay/miniconda3/condabin/conda",
#   pip            = TRUE,
#   pip_ignore_installed = TRUE,
#   pip_options=pip_options
# )
