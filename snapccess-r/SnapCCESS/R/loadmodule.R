#' loadmodule
#'
#' Load all python modules that may used during analysis automatically,
#' including torch, snapccess, and numpy
#'
#' @export
#' @import reticulate
#'
#' @examples #loadmodule()
#'
loadmodule<- function() {
  # use superassignment to update global reference to torch, snapccess, numpy
  torch <- NULL
  snapccess <- NULL
  numpy <- NULL
  if(py_module_available("snapccess")){
    torch <<- reticulate::import("torch", delay_load = TRUE)
    snapccess <<- reticulate::import("snapccess", delay_load = TRUE)
    numpy <<- reticulate::import("numpy",delay_load = TRUE)
  }else{
    cat("Please run install_SnapCCESS")
  }

}
