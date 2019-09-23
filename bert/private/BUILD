
cc_binary(
    name = "prepare_leveldb_dataset",
    srcs = [
        "prepare_leveldb_dataset.cc",
    ],
    copts = [],
    deps = [
        "//train/proto:example_proto_cc",
        "@com_github_google_leveldb//:leveldb",
        "@jsoncpp//:jsoncpp",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_glog//:glog",
    ],
)
