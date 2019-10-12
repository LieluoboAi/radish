package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "scale_product_attention",
    srcs = [
        "scale_product_attention.cc",
    ],
    hdrs = [
        "scale_product_attention.h",
    ],
    deps = [
        "//third_party:pytorch",
    ],
)

cc_library(
    name = "positionwise_fc",
    srcs = [
        "positionwise_fc.cc",
    ],
    hdrs = [
        "positionwise_fc.h",
    ],
    deps = [
        "//radish/layers:layer_norm",
        "//third_party:pytorch",
    ],
)

cc_library(
    name = "multihead_attention",
    srcs = [
        "multihead_attention.cc",
    ],
    hdrs = [
        "multihead_attention.h",
    ],
    deps = [
        ":scale_product_attention",
        "//radish/layers:layer_norm",
        "//radish/utils:logging",
        "//third_party:pytorch",
    ],
)

cc_library(
    name = "encoder_layer",
    srcs = [
        "encoder_layer.cc",
    ],
    hdrs = [
        "encoder_layer.h",
    ],
    deps = [
        ":multihead_attention",
        ":positionwise_fc",
        "//third_party:pytorch",
    ],
)

cc_library(
    name = "decoder_layer",
    srcs = [
        "decoder_layer.cc",
    ],
    hdrs = [
        "decoder_layer.h",
    ],
    deps = [
        ":multihead_attention",
        ":positionwise_fc",
        "//third_party:pytorch",
    ],
)

cc_library(
    name = "transformer_encoder",
    srcs = [
        "transformer_encoder.cc",
    ],
    hdrs = [
        "transformer_encoder.h",
    ],
    deps = [
        ":encoder_layer",
        "//third_party:pytorch",
        "//radish/utils:logging",
    ],
)

cc_library(
    name = "transformer_decoder",
    srcs = [
        "transformer_decoder.cc",
    ],
    hdrs = [
        "transformer_decoder.h",
    ],
    deps = [
        ":decoder_layer",
        "//third_party:pytorch",
    ],
)

cc_library(
    name = "transformer",
    srcs = [
        "transformer.cc",
    ],
    hdrs = [
        "transformer.h",
    ],
    deps = [
        ":transformer_decoder",
        ":transformer_encoder",
        "//third_party:pytorch",
    ],
)
