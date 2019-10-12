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
    ],
)

cc_library(
    name = "example_parser",
    srcs = [
        "example_parser.h",
    ],
    deps = [
        ":llb_example",
        "//third_party:pytorch",
        "//radish/train/proto:example_proto_cc",
        "@jsoncpp//:jsoncpp",
    ],
)

cc_library(
    name = "leveldb_dataset",
    srcs = [
        "leveldb_dataset.h",
    ],
    deps = [
        "//third_party:pytorch",
        "//radish/train/proto:example_proto_cc",
        "//radish/utils:logging",
        "@com_google_absl//absl/strings:strings",
        "@com_github_google_leveldb//:leveldb",
    ],
)

cc_library(
    name = "txt_dataset",
    srcs = [
        "txt_dataset.h",
    ],
    deps = [
        "//third_party:pytorch",
        "//radish/utils:logging",
        "@com_google_absl//absl/strings:strings",
    ],
)
