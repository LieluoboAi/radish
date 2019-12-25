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
        "@spdlog",
    ],
)

cc_library(
    name = "tensor_util",
    srcs = [
        "tensor_util.cc",
        "tensor_util.h",
    ],
    deps = [
        "//third_party:pytorch",
    ],
)

cc_library(
    name = "text_tokenizer",
    srcs = [
        "text_tokenizer.cc",
        "text_tokenizer.h",
    ],
    linkstatic = True,
    deps = [
        ":logging",
    ],
    alwayslink = True,
)

cc_library(
    name = "sentencepiece_tokenizer",
    srcs = [
        "sentencepiece_tokenizer.cc",
        "sentencepiece_tokenizer.h",
    ],
    copts = [],
    linkstatic = True,
    deps = [
        "//radish/utils:logging",
        "//radish/utils:text_tokenizer",
        "@com_google_absl//absl/strings",
        "@sentencepiece//sentencepiece:error",
        "@sentencepiece//sentencepiece:sentencepiece_processor",
    ],
    alwayslink = True,
)

cc_library(
    name="basic_string_util",
    srcs=[
        "basic_string_util.h"
    ]
)