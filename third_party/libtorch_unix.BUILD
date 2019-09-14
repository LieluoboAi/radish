
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
    "-Lexternal/libtorch_unix/lib", 
    "-ltorch",
    "-lc10",
    "-ldl",
  ],
  deps = [

  ],
)