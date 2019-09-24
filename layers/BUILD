package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "embedding",
    srcs = [
        "embedding.cc",
    ],
    hdrs = [
        "embedding.h",
    ],
    deps = [
        "//third_party:pytorch",
    ],
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
        "//utils:logging",
        "//third_party:pytorch",
    ],
)


cc_library(
    name = "modulelist",
    srcs = [
        "modulelist.h",
    ],
    deps = [
        "//third_party:pytorch",
    ],
)