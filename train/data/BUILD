package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "llb_example",
    srcs = [
        "llb_example.h",
    ],
    deps = [
        "//third_party:pytorch",
        "//train/proto:example_proto_cc",
    ],
)

cc_library(
    name = "leveldb_dataset",
    srcs = [
        "leveldb_dataset.h",
    ],
    deps = [
        "//third_party:pytorch",
        "//train/proto:example_proto_cc",
        "@com_google_absl//absl/strings:strings",
        "@com_google_glog//:glog",
        "@com_github_google_leveldb//:leveldb",
    ],
)
