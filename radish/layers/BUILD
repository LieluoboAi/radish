package(
    default_visibility = ["//visibility:public"],
)


cc_library(
    name = "layer_norm",
    srcs = [
        "layer_norm.cc",
    ],
    hdrs = [
        "layer_norm.h",
    ],
    deps = [
        "//radish/utils:logging",
        "//third_party:pytorch",
    ],
)



cc_library(
    name = "crf",
    srcs = [
        "crf.cc",
    ],
    hdrs = [
        "crf.h",
    ],
    deps = [
        "//radish/utils:logging",
        "//third_party:pytorch",
    ],
)