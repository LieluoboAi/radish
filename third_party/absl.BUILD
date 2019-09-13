package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "absl_header",
    hdrs = glob(["absl/*/*.h","absl/*/*/*.h"]),
    includes = [
        ".",
    ],
    visibility = ["//visibility:public"],
    srcs = glob([
        "absl/*/*.h","absl/*/*/*.h"
    ]),
)
