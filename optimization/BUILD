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
        "//utils:logging",
        "@com_google_absl//absl/strings",
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
