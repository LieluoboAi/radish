package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Boost software license

cc_library(
  name = "libtorch",
  hdrs = glob(["include/**/*.h"]),
  includes=[
    "include/",
    "include/torch/csrc/api/include/",
  ],
  linkopts=[
    "-Lexternal/pytorch/lib", 
    "-ltorch",
    "-lc10",
  ],
  deps = [

  ],
)