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
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_glog//:glog",
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
        "//optimization:radam",
        "//third_party:pytorch",
        "//train/data:leveldb_dataset",
    ],
)
