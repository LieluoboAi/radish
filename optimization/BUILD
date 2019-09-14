package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "radam",
    srcs = [
        "radam.cc",
    ],
    hdrs = [
        "radam.h",
    ],
    deps = [
        "//third_party:pytorch",
    ],
)

cc_binary(
    name = "test_radam",
    srcs = ["test_radam.cc"],
    copts = [],
    deps = [
        ":radam",
    ],
)
