package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "span_bert_example",
    srcs = [
        "span_bert_example.cc",
    ],
    hdrs = [
        "span_bert_example.h",
    ],
    deps = [
        "//third_party:pytorch",
        "//train/data:llb_example",
    ],
)

cc_library(
    name = "span_bert_model",
    srcs = [
        "span_bert_model.cc",
    ],
    hdrs = [
        "span_bert_model.h",
    ],
    deps = [
        "//third_party:pytorch",
        "//train:llb_model",
        "//transformer:transformer_encoder",
    ],
)

cc_binary(
    name = "train_span_bert_main",
    srcs = [
        "train_span_bert_main.cc",
    ],
    copts = [],
    deps = [
        ":span_bert_example",
        ":span_bert_model",
        "//train:llb_trainer",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_glog//:glog",
    ],
)
