package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "logging",
    srcs = [
        "logging.h",
    ],
    deps = [
        "@com_google_glog//:glog",
        "@spdlog//:spdlog",
    ],
)

cc_library(
    name = "tensor_util",
    srcs = [
        "tensor_util.h",
         "tensor_util.cc",
    ],
    deps = [
        "//third_party:pytorch",
    ],
)