package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "llb_model",
    srcs = [
        "llb_model.h",
    ],
    deps = [
        "//third_party:pytorch",
    ],
)

cc_library(
    name = "progress_reporter",
    srcs = [
        "progress_reporter.h",
    ],
    deps = [
        "//radish/utils:logging",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
    ],
)

cc_library(
    name = "model_io",
    srcs = [
        "model_io.cc",
        "model_io.h"
    ],
    deps = [
        "//radish/utils:logging",
        "//radish/utils:tensor_util",
        "//third_party:pytorch",
    ],
)
cc_library(
    name = "llb_trainer",
    srcs = [
        "llb_trainer.h",
    ],
    deps = [
        ":progress_reporter",
        ":llb_model",
        ":model_io",
        "//radish/utils:logging",
        "//radish/optimization:radam",
        "//radish/optimization:lamb",
        "//third_party:pytorch",
        "//radish/train/data:leveldb_dataset",
        "//radish/train/data:txt_dataset",
        "@gulrak_filesystem//:filesystem",
    ],
)
